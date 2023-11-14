# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from collections import namedtuple
import math
import torch
import torch.nn as nn
from fairseq import utils, tasks, checkpoint_utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text import (
    TransformerDecoder,
    Conv1dSubsampler,
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    GradMultiply,
    PositionalEmbedding,
    FairseqDropout,
)
from fairseq.data.data_utils import lengths_to_padding_mask
from typing import Dict, List
from torch import Tensor
import sys
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DualInputEncoder(FairseqEncoder):
    def __init__(
        self,
        args,
        dictionary,
        text_encoder,
        cross_attentive_loss_before_last_layer=-1,
    ):
        super().__init__(dictionary)
        self.cross_attentive_loss_before_last_layer = (
            cross_attentive_loss_before_last_layer
        )
        self.use_cross_attentive_loss = (
            False if cross_attentive_loss_before_last_layer <= -1 else True
        )
        self.token_type_embeddings = nn.Embedding(2, args.encoder_embed_dim)
        self.padding_idx = 1
        self.asr_idx = dictionary.index("<asr>")
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
        )
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0

        self._build_acoustic_encoder(args)
        self.text_encoder = text_encoder


    def _build_acoustic_encoder(self, args):
        assert args.load_pretrain_audio_encoder is not None
        self.w2v2_model_path = args.load_pretrain_audio_encoder
        # self.use_asr_finetune_w2v = args.use_asr_finetune_w2v
        self.wav2vec_model, self.w2v_args, w2v_task = checkpoint_utils.load_model_ensemble_and_task([args.load_pretrain_audio_encoder])
        self.wav2vec_model=self.wav2vec_model[0]
        self.freeze_w2v = args.freeze_w2v
        self.subsample_audio = Conv1dSubsampler(
            self.wav2vec_model.post_extract_proj.out_features,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):
        cfg = {
            "encoder_embed_dim": args.encoder_text_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.text_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "encoder_learned_pos": args.encoder_learned_pos,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "adaptive_input": args.adaptive_input,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
        }
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        enc_emb = nn.Embedding(
            len(src_dictionary), model_args.encoder_embed_dim, src_dictionary.pad()
        )
        text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)
        return text_encoder

    def mult_rst_grad(self, rst, ratio):
        assert isinstance(rst, dict)  # instead of EncoderOut
        assert len(rst["encoder_out"]) == 1
        rst["encoder_out"][0] = GradMultiply.apply(rst["encoder_out"][0], ratio)
        return rst

    def process_attentive_loss_states(self, rst, interstates):
        assert isinstance(rst, dict)  # instead of EncoderOut
        rst["encoder_states"] = interstates
        return rst

    def _get_w2v_padding_mask(self, w2v_feature, src_lengths):
        output_lengths = self.wav2vec_model._get_feat_extract_output_lengths(src_lengths)
        padding_mask = torch.zeros(w2v_feature.shape[:2], dtype=w2v_feature.dtype, device=w2v_feature.device)
        padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,)] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return padding_mask

    def _get_w2v_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        wav2vec_out= self.wav2vec_model.extract_features(src_tokens, padding_mask)
        w2v_feature = wav2vec_out["x"]
        padding_mask = self._get_w2v_padding_mask(w2v_feature, src_lengths)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return w2v_feature, padding_mask, output_length # T*B*C

    def embedding_audio(self, src_tokens, src_lengths,
                        return_short_audio_len=False):
        w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
            src_tokens, src_lengths)

        x, input_lengths = self.subsample_audio(w2v_feature, input_lengths)
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout_module(x)
        x = x + self.token_type_embeddings(torch.ones(x.shape[0:2], dtype=int, device=x.device)) 
        if return_short_audio_len:
            return x, encoder_padding_mask, input_lengths
        return x, encoder_padding_mask

    def embedding_text(self, src_txt_tokens, src_lengths):
        x, _ = self.text_encoder.forward_embedding(src_txt_tokens)  # B*T*C
        x = x + self.token_type_embeddings(torch.zeros(x.shape[0:2], dtype=int, device=x.device))
        x = x.transpose(0, 1)
        encoder_padding_mask = src_txt_tokens.eq(self.padding_idx)
        return x, encoder_padding_mask #  T x B x C, B*T

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        src_txt_tokens=None,
        src_txt_lengths=None,
        test=None,
        **kwargs
    ):
        """
        Args:
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (speech) (B,)
            src_txt_tokens: padded tensor (B, T)
            src_txt_lengths: tensor of original lengths of input utterances (text) (B,)
        """
        # src_tokens only: inference
        # src_tokens, src_lengths: speech only training
        # src_txt_tokens, src_txt_lengths: text only training
        # all valid: speech + text training
        if src_tokens is None and src_txt_tokens is None:
            raise ValueError(
                "src_tokens and src_txt_tokens cannot be None at the same time"
            )
        ret_audio = None
        ret_text = None
        return_all_hiddens = False
        if test == 'speech':
            src_txt_tokens = None
        elif test == 'text':
            src_tokens = None
        elif test == 'mt':
            src_txt_tokens, src_txt_lengths = src_tokens, src_txt_lengths
            src_tokens, src_txt_lengths = None, None

        if src_txt_tokens is not None:
            if src_tokens is not None:
                if self.use_cross_attentive_loss:
                    return_all_hiddens = True

                text_embeds, text_mask = self.embedding_text(src_txt_tokens, src_txt_lengths)
                audio_embeds, audio_mask = self.embedding_audio(src_tokens, src_lengths)

                co_embeds = torch.cat([audio_embeds, text_embeds], dim=0)
                co_lengths = torch.cat([audio_mask, text_mask], dim=1)

                ret_fuse = self.text_encoder.forward_straight(
                    co_embeds, co_lengths, return_all_hiddens=return_all_hiddens
                )   

                if not self.training and test is not None:
                    return ret_fuse

                ret_audio = self.text_encoder.forward_straight(
                    audio_embeds, audio_mask, return_all_hiddens=return_all_hiddens
                )

                ret_text = self.text_encoder.forward_straight(
                    text_embeds, text_mask, return_all_hiddens=return_all_hiddens
                )

                return (ret_audio, ret_text, ret_fuse)
            else:
                text_embeds, text_mask = self.embedding_text(src_txt_tokens, src_txt_lengths)
                ret_text = self.text_encoder.forward_straight(text_embeds, text_mask, return_all_hiddens=return_all_hiddens)
                return ret_text
        else:
            if src_tokens is not None:
                audio_embeds, audio_mask = self.embedding_audio(src_tokens, src_lengths)
                ret_audio = self.text_encoder.forward_straight(audio_embeds, audio_mask, return_all_hiddens=return_all_hiddens)
                return ret_audio

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": None,  # B x T x C
            "encoder_states": None,  # List[T x B x C]
            "src_tokens": None,  # B x T
            "src_lengths": None,  # B x 1
        }

# TransformerMultiInputDecoder: take one or two encoder inputs
class TransformerMultiInputDecoder(FairseqDecoder):
    def __init__(
        self,
        dictionary,
        text_decoder,
        asr_decoder,
        compute_cross_attentive_loss=False,
        cross_attentive_loss_with_norm=True,
        cross_attentive_loss_reverse=True,
    ):

        super().__init__(dictionary)
        self.text_decoder = text_decoder
        self.asr_decoder = asr_decoder
        self.compute_cross_attentive_loss = compute_cross_attentive_loss
        self.cross_attentive_loss_with_norm = cross_attentive_loss_with_norm
        self.cross_attentive_loss_reverse = cross_attentive_loss_reverse

    def cross_attentive_loss(
        self, teacher_states, student_states, teacher_masking, student_masking, eps=1e-6
    ):
        x = teacher_states.transpose(0, 1)  # from T X B X D to B X T X D
        y = student_states.transpose(0, 1)
        if self.cross_attentive_loss_with_norm:
            x = x / (x.norm(dim=2, keepdim=True) + eps)
            y = y / (y.norm(dim=2, keepdim=True) + eps)
        dim = x.size(-1)
        # lengths: batch X seqLen
        sim_scores_xy = torch.bmm(x, y.transpose(1, 2))  # batch X lenx X leny ]
        if y.dtype == torch.float16:
            sim_scores_xy = sim_scores_xy.float()
            y = y.float()
            x = x.float()
        if teacher_masking != []:
            assert len(teacher_masking) == 1
            sim_scores_xy = sim_scores_xy.masked_fill(
                teacher_masking[0].unsqueeze(-1), float("-inf")
            )
        if student_masking != []:
            sim_scores_xy = sim_scores_xy.masked_fill(
                student_masking[0].unsqueeze(1), float("-inf")
            )
        # do masking
        y_weights = utils.softmax(sim_scores_xy, dim=-1)
        if teacher_masking != []:
            y_weights = y_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)
        x_reconstruct_from_y = torch.bmm(y_weights, y)

        sim_scores_xx = torch.bmm(x, x.transpose(1, 2))  # batch X lenx X lenx ]
        x_weights = utils.softmax(sim_scores_xx, dim=-1)
        if teacher_masking != []:
            x_weights = x_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)

        # no gradient for teacher state
        x_reconstruct_from_x = torch.bmm(x_weights, x).detach()
        cost = (x_reconstruct_from_x - x_reconstruct_from_y).norm(dim=2)
        if teacher_masking != []:
            cost = cost.masked_fill(teacher_masking[0], 0)

        if not self.cross_attentive_loss_with_norm:
            cost = cost / dim
        return cost

    def asr_forward(
        self,
        prev_output_asr_tokens,
        encoder_out,
        incremental_state=None,
        **kwargs
    ):
        return self.asr_decoder(prev_output_asr_tokens, encoder_out, incremental_state)

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        has_txt_input=False,
        **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing. If there are
                two or more input during training, they will share the same prev_output_tokens
            encoder_out (tuple[Tensor]): output from the encoder, used for
                encoder-side attention. It will be tuple if there are more inputs, but a tensor
                if only one input
            incremental_state ([dict]): dictionary used for storing state during
                :ref:`Incremental decoding`. It is only valid for inference, only from single
                input
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`. If there are N inputs, batch will be N bigger than a single input
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        assert not isinstance(encoder_out, EncoderOut)
        if isinstance(encoder_out, tuple):  # training with mulitple input
            rst = []
            assert len(encoder_out) == 3
            for i, eo in enumerate(encoder_out):
                assert incremental_state is None
                rst.append(self.text_decoder(prev_output_tokens, eo, incremental_state))
            dec_out = torch.cat([r[0] for r in rst], dim=0)
            attn_cost = None
            if self.compute_cross_attentive_loss:
                assert isinstance(encoder_out[0], dict)
                if self.cross_attentive_loss_reverse: # use asr data
                    attn_cost1 = self.cross_attentive_loss(
                        teacher_states=encoder_out[2]["encoder_states"], 
                        student_states=encoder_out[0]["encoder_states"], 
                        teacher_masking=encoder_out[2]["encoder_padding_mask"],
                        student_masking=encoder_out[0]["encoder_padding_mask"],
                    )     
                    attn_cost2 = self.cross_attentive_loss(
                        teacher_states=encoder_out[2]["encoder_states"],
                        student_states=encoder_out[1]["encoder_states"],  
                        teacher_masking=encoder_out[2]["encoder_padding_mask"],
                        student_masking=encoder_out[1]["encoder_padding_mask"],
                    )
                else: # not use asr data
                    attn_cost1 = self.cross_attentive_loss(
                        teacher_states=encoder_out[1]["encoder_states"], 
                        student_states=encoder_out[0]["encoder_states"],  
                        teacher_masking=encoder_out[1]["encoder_padding_mask"],
                        student_masking=encoder_out[0]["encoder_padding_mask"],
                    )     
                    attn_cost2 = self.cross_attentive_loss(
                        teacher_states=encoder_out[1]["encoder_states"], 
                        student_states=encoder_out[2]["encoder_states"], 
                        teacher_masking=encoder_out[1]["encoder_padding_mask"],
                        student_masking=encoder_out[2]["encoder_padding_mask"],
                    )
                attn_cost = attn_cost1 + attn_cost2      
            return (dec_out, {"attn_cost": attn_cost})
        else:  # inference or training with one input
            if has_txt_input:
                return self.text_decoder(
                    prev_output_tokens, encoder_out, incremental_state
                )
            return self.text_decoder(prev_output_tokens, encoder_out, incremental_state)


# Note:
# dual input transformer:
#    encoder: S2TTransformerEncoder for speech + TransformerEncoder for text
#    decoder: TransformerDecoder for text
@register_model("fstnet")
class DualInputS2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    def max_positions(self):
        return None  # it is provided in task

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # encoder 1: S2TTransformerEncoder for speech
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="""
                encoder output dimension, can be None. If specified, projecting the
                transformer output to the specified dimension""",
        )
        # standard Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-text-embed-dim",
            type=int,
            metavar="N",
            help="encoder text embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        # non-standard transformer parameters
        parser.add_argument(
            "--speech-encoder-layers",
            type=int,
            metavar="N",
            help="num speech encoder layers",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            metavar="N",
            help="num text encoder layers",
        )
        ###
        parser.add_argument(
            "--text-input-cost-ratio",
            type=float,
            default=1.0,
            metavar="V",
            help="text input cost ratio relative to speech input cost",
        )
        parser.add_argument(
            "--init-scale",
            type=float,
            default=1.0,
            metavar="V",
            help="scale the initial weight by given factor",
        )
        parser.add_argument(
            "--load-pretrain-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained encoder """,
        )
        parser.add_argument(
            "--load-pretrain-audio-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech encoder """,
        )
        parser.add_argument(
            "--use-asr-finetune-w2v",
            type=bool,
            default=True,
        )
        parser.add_argument(
            "--freeze-w2v",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--load-pretrain-speech-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder-last",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-decoder",
            type=str,
            metavar="EXPR",
            default="",
            help=""" path to the pretrained encoder """,
        )
        parser.add_argument(
            "--add-speech-eos",
            action="store_true",
            help="add eos token at the end of input feature",
        )
        parser.add_argument(
            "--speech-encoder-adapter-type",
            type=str,
            metavar="EXPR",
            default="None",
            choices=["None", "Linear", "MLP"],
            help="add speech encoder adapter",
        )
        parser.add_argument(
            "--use-ctc",
            default=False,
            action="store_true",
            help="add ctc projection after w2v encoder"
        )

    @classmethod
    def build_encoder(cls, args, task):
        text_encoder = DualInputEncoder.build_text_encoder(
            args, task.src_dict
        )
        cross_attentive_loss_before_last_layer = (
            0 if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else -1
        )
        encoder = DualInputEncoder(
            args,
            task.src_dict,
            text_encoder,
            cross_attentive_loss_before_last_layer,
        )
        if args.init_scale != 1.0:
            with torch.no_grad():
                for param in encoder.parameters():
                    param.data.mul_(args.init_scale)
        if args.load_pretrain_text_encoder != "":
            checkpoint_utils.load_pretrained_component_from_model(
                text_encoder, args.load_pretrain_text_encoder
            )
        if (
            args.load_pretrain_text_encoder_last != ""
        ):  # if share encoder, speech encoder parameters will be used.
            # It provides a chance to use pre-trained mt encoder instead
            checkpoint_utils.load_pretrained_component_from_model(
                text_encoder, args.load_pretrain_text_encoder_last
            )

        if args.load_pretrain_encoder != "":
            checkpoint_utils.load_pretrained_component_from_model(
                encoder, args.load_pretrain_encoder
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        dec_cfg = {
            "decoder_layerdrop": args.decoder_layerdrop,
            "share_decoder_input_output_embed": args.share_decoder_input_output_embed,
            "decoder_embed_dim": args.decoder_embed_dim,
            "max_target_positions": args.max_target_positions,
            "dropout": args.dropout,
            "encoder_learned_pos": args.encoder_learned_pos,
            "decoder_learned_pos": args.decoder_learned_pos,
            "layernorm_embedding": args.layernorm_embedding,
            "decoder_normalize_before": args.decoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "decoder_ffn_embed_dim": args.decoder_ffn_embed_dim,
            "decoder_layers": args.decoder_layers,
            "decoder_attention_heads": args.decoder_attention_heads,
            "decoder_output_dim": args.decoder_embed_dim,
            "no_scale_embedding": args.no_scale_embedding,
            "adaptive_input": args.adaptive_input,
            "quant_noise_pq": args.quant_noise_pq,
            "adaptive_softmax_cutoff": args.adaptive_softmax_cutoff,
            "tie_adaptive_weights": args.tie_adaptive_weights,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "encoder": {"embed_dim": args.encoder_embed_dim}
        }
        dec_cfg = namedtuple("args", dec_cfg.keys())(*dec_cfg.values())
        dec_emb = nn.Embedding(
            len(task.target_dictionary),
            args.decoder_embed_dim,
            task.target_dictionary.pad(),
        )
        compute_cross_attentive_loss = (
            True if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else False
        )
        cross_attentive_loss_without_norm = getattr(
            args, "attentive_cost_without_normalize", False
        )
        cross_attentive_loss_reverse = (
            True  # getattr(args, "attentive_cost_reverse", False)
        )

        text_decoder = TransformerDecoder(dec_cfg, task.target_dictionary, dec_emb)
        if args.use_asr_finetune_w2v:
            asr_decoder = TransformerDecoder(dec_cfg, task.src_dict, dec_emb)
        else:
            asr_decoder = None
        decoder = TransformerMultiInputDecoder(
            dictionary=task.target_dictionary,
            text_decoder=text_decoder,
            asr_decoder=asr_decoder,
            compute_cross_attentive_loss=compute_cross_attentive_loss,
            cross_attentive_loss_with_norm=True
            if not cross_attentive_loss_without_norm
            else False,
            cross_attentive_loss_reverse=cross_attentive_loss_reverse,
        )
        if args.init_scale != 1.0:
            with torch.no_grad():
                for param in decoder.parameters():
                    param.data.mul_(args.init_scale)
        if args.load_pretrain_decoder != "":
            try:
                checkpoint_utils.load_pretrained_component_from_model(
                    decoder, args.load_pretrain_decoder
                )
            except RuntimeError:
                checkpoint_utils.load_pretrained_component_from_model(
                    decoder.text_decoder, args.load_pretrain_decoder
                )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        dualinputs2ttransformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        use_encoder_outputs=False,
        src_txt_tokens=None,
        src_txt_lengths=None,
        mode="sup_speech",
        **kwargs
    ):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            mode = 'sup_speech' or 'text'

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if mode == "text":
            assert src_txt_tokens is None
            src_txt_tokens = src_tokens
            src_txt_lengths = src_lengths
            src_tokens = None
            src_lengths = None
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths,
            **kwargs
        )
        has_txt_input = True if src_txt_tokens is not None else False
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            has_txt_input=has_txt_input,
            **kwargs
        )
        if use_encoder_outputs:
            return decoder_out, encoder_out
        return decoder_out


@register_model_architecture(
    "fstnet", "fstnet_base"
)
def dualinputs2ttransformer_base(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_text_embed_dim = getattr(
        args, "encoder_text_embed_dim", args.encoder_embed_dim
    )
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 10)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.add_speech_eos = getattr(args, "add_speech_eos", False)


@register_model_architecture("fstnet", "fstnet_m")
def dualinputs2ttransformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)


@register_model_architecture("fstnet", "fstnet_b")
def dualinputs2ttransformer_b(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.15)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)


@register_model_architecture("fstnet", "fstnet_24E6D")
def dualinputs2ttransformer_b(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.15)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 24)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 24)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)


@register_model_architecture("fstnet", "fstnet_l")
def dualinputs2ttransformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)
