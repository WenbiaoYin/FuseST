# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq import metrics, utils


@register_criterion("mutil_task_for_fuse_speech_text")
class GuidedCrossEntAccCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        guide_alpha,
        text_input_cost_ratio,
        label_smoothing,
        kl_st=False,
        ctc_weight=0.0,
        jsd_weight=1.0,
        contrastive_weight=0.0,
        contrastive_temperature=1.0,
        ignore_prefix_size=0,
        use_dual_ctr=False,
        disable_text_guide_update_num=0,
        attentive_cost_regularization=0,
    ):
        """
        guide_alpha:            alpha to inteplate nll and kd loss
        text_input_cost_ratio:  loss ratio for text only input data
        label_smoothing:        label smoothing ratio
        disable_text_guide_update_num:  only use nll loss for the first N updates
        attentive_cost_regularization:  ratio fo attentive cost
        """
        super().__init__(task)
        self.alpha = guide_alpha
        self.attn_beta = attentive_cost_regularization
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.padding_idx = 1
        self.text_input_cost_ratio = text_input_cost_ratio
        self.disable_update_num = disable_text_guide_update_num
        assert self.alpha >= 0 and self.alpha <= 1.0
        self.ctc_weight = ctc_weight
        self.jsd_weight = jsd_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.ignore_prefix_size = ignore_prefix_size
        self.use_dual_ctr = use_dual_ctr
        self.kl_st = kl_st
        self.zero_infinity = True
        self.blank_idx, self.pad_idx, self.eos_idx = 0, 1, 2

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        FairseqCriterion.add_args(parser)
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: off
        parser.add_argument('--guide-alpha', default=0., type=float, metavar='D',
                            help='alpha to merge kd cost from text to speech input with ce loss')
        # fmt: off
        parser.add_argument('--disable-text-guide-update-num', default=0, type=int, metavar='D',
                            help='disable guided target from text for the first N updates.')
        parser.add_argument("--attentive-cost-regularization", default=0.0, type=float, metavar='D',
                            help="use encoder attentive loss regularization with cost ratio D")
        parser.add_argument("--attentive-cost-without-normalize", action='store_true',
                            help="Don't do normalization during attentive cost computation")
        parser.add_argument("--ctc-weight", default=0.0, type=float,
                            help="weight for ctc loss")
        parser.add_argument("--jsd-weight", default=1.0, type=float,
                            help="weight for jsd loss")
        parser.add_argument("--contrastive-weight", default=0.0, type=float,
                            help="weight for contrastive loss")
        parser.add_argument("--contrastive-temperature", default=0.02, type=float,
                            help="temperature for contrastive loss") 
        parser.add_argument("--use-dual-ctr", action="store_true",
                            help="if we want to use dual contrastive loss")
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')                                      

    def compute_jsd_loss(self, lprobs_st, lprobs_mix, target, ignore_index):
        kl_loss_st = F.kl_div(lprobs_mix, lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_loss_mix = F.kl_div(lprobs_st, lprobs_mix, log_target=True, reduction="none").sum(-1)
        pad_mask = target.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        kl_loss_mix.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mix = kl_loss_mix.sum()
        kl_loss = (kl_loss_st + kl_loss_mix) / 2.0
        return kl_loss

    def forward(self, model, sample, reduce=True):
        reduction = 'sum' if reduce else 'none'
        net_input = sample["net_input"]
        if self.contrastive_weight > 0.0 or self.ctc_weight > 0.0 :
            net_input["use_encoder_outputs"]=True
            net_output, encoder_out = model(**net_input)
        else:
            net_output = model(**net_input)
        attn_cost, jsd_loss, ctc_loss, ctc_nll_loss, contrastive_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        is_dual_input = True if net_input['src_tokens'] is not None and net_input.get('src_txt_tokens') is not None else False
        target = model.get_targets(sample, net_output)
        src_token_num, text_token_num = 0, 0
        if is_dual_input:
            # lprobs_spch from speech encoder and lprobs_text from text encoder
            lprobs_spch, lprobs_text, lprobs_text_speech= torch.chunk(lprobs, 3)
            lprobs_spch.batch_first = lprobs.batch_first
            lprobs_text.batch_first = lprobs.batch_first
            lprobs_text_speech.batch_first = lprobs.batch_first

            speech_loss, speech_nll_loss, speech_correct, speech_total = \
                self.guide_loss_and_acc(model, lprobs_spch, lprobs_text_speech, target, reduce=(reduction == 'sum'))
            text_loss, text_nll_loss, text_correct, text_total = \
                self.guide_loss_and_acc(model, lprobs_text, lprobs_text_speech, target, reduce=(reduction == 'sum'))
            text_speech_loss, text_speech_nll_loss, text_speech_correct, text_speech_total = \
                self.compute_loss_and_acc(model, lprobs_text_speech, target, reduction=reduction)

            jsd_loss = (self.compute_jsd_loss(lprobs_text_speech, lprobs_spch, target, self.padding_idx) + \
                self.compute_jsd_loss(lprobs_text_speech, lprobs_text,target, self.padding_idx)) / 2.0

            text_token_num = target.ne(self.padding_idx).sum() 
            attn_cost = net_output[1].get('attn_cost')
            if attn_cost is not None:
                # attn_cost is batch_first and padding tokens have been masked already
                src_token_num = attn_cost.ne(0).sum()
                attn_cost = attn_cost.sum()
            else:
                attn_cost = torch.tensor(0.0)
            if self.ctc_weight > 0.0:
                ctc_loss, ctc_nll_loss = self.compute_loss_asr( model, sample, encoder_out[0], reduce=True)
            if self.contrastive_weight > 0.0:
                contrastive_loss=self.compute_contrastive_loss( model, net_input, encoder_out[0], reduce=True )

            loss = speech_loss + text_loss + text_speech_loss + self.jsd_weight * jsd_loss + \
                 attn_cost * self.attn_beta + self.ctc_weight * ctc_loss + self.contrastive_weight * contrastive_loss
            nll_loss = speech_nll_loss + text_nll_loss + text_speech_nll_loss + self.ctc_weight * ctc_nll_loss
            correct = speech_correct + text_correct + text_speech_correct
            total = speech_total + text_total + text_speech_total    
        else:
            loss, nll_loss, correct, total = self.compute_loss_and_acc(model, lprobs, target, reduction=reduction)
            if sample["net_input"]['src_tokens'] is None:   # text input only
                loss = loss * self.text_input_cost_ratio
            speech_loss, speech_nll_loss = None, None
            text_loss, text_nll_loss =  None, None
            text_speech_loss, text_speech_nll_loss = None, None

        sample_size, logging_output = self.get_logging_output(
            sample, loss, nll_loss, correct, total, src_token_num, text_token_num, jsd_loss, ctc_loss, contrastive_loss, speech_loss, speech_nll_loss, text_loss, text_nll_loss, text_speech_loss, text_speech_nll_loss, attn_cost, is_dual_input
        )
        return loss, sample_size, logging_output

    def compute_loss_and_acc(self, model, lprobs, target, reduction='sum'):
        if not lprobs.batch_first:
            lprobs = lprobs.transpose(0, 1)
        lprobs = lprobs.view(-1, lprobs.size(-1))  # -> (B x T) x C
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=(reduction == 'sum'),
        )

        mask = target.ne(self.padding_idx)
        correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total

    def guide_loss_and_acc(self, model, lprobs, lprobs_teacher, target, reduce=True):
        """ lprobs_teacher is used as guide for lprobs """
        if self.alpha == 0.0 or model.num_updates < self.disable_update_num:
            return self.compute_loss_and_acc(model, lprobs, target, reduction=('sum' if reduce else 'none'))
        if not lprobs.batch_first:
            lprobs = lprobs.transpose(0, 1)
            lprobs_teacher = lprobs_teacher.transpose(0, 1)

        lprobs = lprobs.view(-1, lprobs.size(-1)).float()  # -> (B x T) x C
        lprobs_teacher = lprobs_teacher.view(-1, lprobs_teacher.size(-1)).float()  # -> (B x T) x C
        target = target.view(-1)
        loss = F.nll_loss(lprobs, target, ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')
        nll_loss = loss
        probs_teacher = lprobs_teacher.exp().masked_fill_(target.unsqueeze(-1).eq(self.padding_idx), 0)
        probs_teacher = probs_teacher.detach()
        guide_loss = -(probs_teacher*lprobs).sum() if reduce else -(probs_teacher*lprobs).sum(-1, keepdim=True)
        loss = self.alpha*guide_loss + (1.0 - self.alpha)*loss

        mask = target.ne(self.padding_idx)
        correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total

    def compute_loss_asr(self, model, sample, encoder_out, reduce=True):
        net_output = model.decoder.asr_forward(sample["prev_output_asr_tokens"],encoder_out)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["asr_target"]
        if target[0]=="":
            return torch.tensor(0.0), torch.tensor(0.0)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def get_sequence_hidden(self, model, sample, packed_encoder_out, is_text=False):
        if is_text:
            encoder_out, encoder_padding_mask = model.encoder.embedding_text(
                sample["src_txt_tokens"], sample["src_txt_lengths"])
        else:
            encoder_out = packed_encoder_out["encoder_embedding"][0]
            encoder_padding_mask = packed_encoder_out["encoder_padding_mask"][0]
        encoder_out = encoder_out.transpose(0, 1) # T x B x hid -> B x T x hid
        encoder_padding_mask = (~encoder_padding_mask).float()
        seq_hidden = (encoder_out * encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / encoder_padding_mask.sum(dim=1).unsqueeze(-1)
        return seq_hidden

    def compute_contrastive_loss(self, model, sample, encoder_out, reduce=True):
        audio_seq_hidden = self.get_sequence_hidden(model, sample, encoder_out, is_text=False,) # B x h
        text_seq_hidden = self.get_sequence_hidden(model, sample, encoder_out, is_text=True) # B x h

        batch_size, hidden_size = audio_seq_hidden.size()
        logits = F.cosine_similarity(audio_seq_hidden.expand((batch_size, batch_size, hidden_size)),
                                     text_seq_hidden.expand((batch_size, batch_size, hidden_size)).transpose(0, 1),
                                     dim=-1)
        logits /= self.contrastive_temperature

        if self.use_dual_ctr:
            loss_audio = -torch.nn.LogSoftmax(0)(logits).diag()
            loss_text = -torch.nn.LogSoftmax(1)(logits).diag()
            loss = loss_audio + loss_text
        else:
            loss = -torch.nn.LogSoftmax(0)(logits).diag()

        if reduce:
            loss = loss.sum()
        return loss

    def get_logging_output(
        self,
        sample,
        loss,
        nll_loss,
        correct,
        total,
        src_token_num=0,
        text_token_num=0,
        jsd_loss = None,
        ctc_loss = None,
        contrastive_loss=None,
        speech_loss=None,
        speech_nll_loss=None,
        text_loss=None,
        text_nll_loss=None, 
        text_speech_loss=None, 
        text_speech_nll_loss=None,
        attn_cost=None,
        is_dual_input=False,
    ):

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        mul_size = 2 if is_dual_input else 1

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "nll_loss": utils.item(nll_loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"]*mul_size,
            "nsentences": sample["target"].data.size(0)*mul_size,
            "sample_size": sample_size*mul_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "src_token_num": utils.item(src_token_num.data) if src_token_num > 0 else 0,
            "text_token_num": utils.item(text_token_num.data) if text_token_num > 0 else 0,
            "nframes": torch.sum(sample["net_input"]["src_lengths"].data).item(),
        }

        if speech_loss is not None:
            logging_output["speech_loss"] = utils.item(speech_loss.data)
            logging_output["speech_nll_loss"] = utils.item(speech_nll_loss.data)
            logging_output["sample_size_speech_cost"] = sample_size
            logging_output["speech_attn_loss"] = attn_cost.data

        if jsd_loss is not None:
            logging_output["jsd_loss"] = utils.item(jsd_loss.data)

        if  ctc_loss is not None:
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)
        
        if contrastive_loss is not None:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)

        if text_loss is not None:
            logging_output["text_loss"] = utils.item(text_loss.data)
            logging_output["text_nll_loss"] = utils.item(text_nll_loss.data)

        if text_speech_loss is not None:
            logging_output["text_speech_loss"] = utils.item(text_speech_loss.data)
            logging_output["text_speech_nll_loss"] = utils.item(text_speech_nll_loss.data)

        return sample_size*mul_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        src_token_sum = sum(log.get("src_token_num", 0) for log in logging_outputs)
        text_token_sum = sum(log.get("text_token_num", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        speech_loss_sum = sum(log.get("speech_loss", 0) for log in logging_outputs)
        speech_nll_loss_sum = sum(log.get("speech_nll_loss", 0) for log in logging_outputs)
        text_loss_sum = sum(log.get("text_loss", 0) for log in logging_outputs)
        text_nll_loss_sum = sum(log.get("text_nll_loss", 0) for log in logging_outputs)
        text_speech_loss_sum = sum(log.get("text_speech_loss", 0) for log in logging_outputs)
        text_speech_nll_loss_sum = sum(log.get("text_speech_nll_loss", 0) for log in logging_outputs)
        speech_attn_loss_sum = sum(log.get("speech_attn_loss", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        sample_size_speech = sum(log.get("sample_size_speech_cost", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            "nll_loss": nll_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, and loss
            # is per-sentence loss; else sample_size is ntokens, and the loss
            # becomes per-output token loss
            "speech_loss": speech_loss_sum / sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "speech_nll_loss": speech_nll_loss_sum / sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "text_loss": text_loss_sum / text_token_sum / math.log(2) if text_token_sum > 0 else 0.0,
            "text_nll_loss": text_nll_loss_sum / text_token_sum / math.log(2) if text_token_sum > 0 else 0.0,
            "text_speech_loss": text_speech_loss_sum /  sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "text_speech_nll_loss": text_speech_nll_loss_sum / sample_size_speech / math.log(2) if sample_size_speech > 0 else 0.0,
            "speech_attn_loss": speech_attn_loss_sum / src_token_sum / math.log(2) if src_token_sum > 0 else 0.0,
            "jsd_loss": jsd_loss_sum / src_token_sum / math.log(2) if src_token_sum > 0 else 0.0,
            "ctc_loss": ctc_loss_sum / src_token_sum / math.log(2) if src_token_sum > 0 else 0.0,
            "contrastive_loss": contrastive_loss_sum / nsentences / math.log(2) if nsentences > 0 else 0.0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            "src_token_num": src_token_sum,
            # total is the number of validate tokens
        }
        return agg_output

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v, round=3)
