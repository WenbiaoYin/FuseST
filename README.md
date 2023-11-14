# FST: Improving speech translation by fusing speech and text

## Overview





## Download Trained Models

The models are trained based on fairseq. You may download all the models at Google drive.



| **Datasets** |                     **Model Checkpoint**                     |                       **SPM & Vocab**                        | FST-MT BLUE | FST-PT BLUE |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------: | :---------: |
|    En-De     | [Download](https://drive.google.com/file/d/1M8wZtsEgaspI0vRgZinaZ0fi-yDXuL1a/view?usp=share_link) | [spm](https://drive.google.com/file/d/1vsHHQ9w3U6g4Ipo1xDq1rH45P1gV0-NR/view?usp=share_link) & [vocab](https://drive.google.com/file/d/1PfOsFhmFc37Iv6m0TKmnNyCR0I5FaUGI/view?usp=share_link) |    27.7     |    29.2     |
|    En-Es     | [Download](https://drive.google.com/file/d/1rJS5olcntThH1DHbE_eb5dk4fCW09x97/view?usp=share_link) | [spm](https://drive.google.com/file/d/1pqvyAHCllhXbYU8I5EOGEM4Wj1wU7zQ7/view?usp=share_link) & [vocab](https://drive.google.com/file/d/1xflLhjmfsoXnoTBs6nuslHeQu5giRLqC/view?usp=share_link) |    32.4     |    33.9     |
|    En-Fr     | [Download](https://drive.google.com/file/d/1ePSSf9FCe58qlVyNi4pxty_zBB61gcp_/view?usp=share_link) | [spm](https://drive.google.com/file/d/1pqvyAHCllhXbYU8I5EOGEM4Wj1wU7zQ7/view?usp=share_link) & [vocab](https://drive.google.com/file/d/1IzImvpJTjIL4tFI_fnV6Ge30neLD3Fb0/view?usp=share_link) |    37.2     |    38.9     |
|    En-Zh     | [Download](https://drive.google.com/file/d/1oWIHufvk4u2mIq3PfLx_gpBYz-LcWacD/view?usp=share_link) | [spm](https://drive.google.com/file/d/1PfOsFhmFc37Iv6m0TKmnNyCR0I5FaUGI/view?usp=share_link) & [vocab](https://drive.google.com/file/d/1rA6gfL4IDbVUd1vaCXbJ1FdHUDxmRBjQ/view?usp=share_link) |      -      |      -      |



## Requirements and Installation

- [PyTorch](http://pytorch.org/) version >= 1.10.0
- Python version >= 3.8

- For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

```shell
git clone git@github.com:WenbiaoYin/FST.git
cd FST
pip install --editable ./
```



## Data Preparation

### MuST-C Data Preparation

1. Download the [MuST-C v1.0](https://ict.fbk.eu/must-c/) to the `${DATA_PATH}` , and uncompress it:

```shell
cd ${DATA_PATH}
tar -xzvf MUSTC_v1.0_en-de.tar.gz
```

2. Then run the following script to generate the yaml configuration file, tsv file, sub-word model and dictionary. In this work, We jointly tokenize the bilingual text (En & X) using [SentencePiece](https://github.com/google/sentencepiece), with a vocabulary size of 10k. For example,

```

python3 FST/examples/speech_to_text/prep_mustc_data.py --data-root ${DATA_PATH} --lang de --vocab-type bpe  --vocab-size 16000

```

3. Finally, the directory `$MUSTC_ROOT` should look like this:

```
data
├── en-de
│   └── *.wav of train/dev/tst-COMMON...
├── config.yaml
├── dev.tsv
├── spm_bpe16000.model
├── spm_bpe16000.txt
├── spm_bpe16000.vocab
├── train.tsv
├── tst-COMMON.tsv
├── tst-HE.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

### MT Data Preparation

You can use  [SentencePiece](https://github.com/google/sentencepiece)  to pre-process the extra MT data, and save the parallel data under `${PARALLEL_MT_DATA}`

```
spm_encode --model=spm_bpe16000.model --output_format=piece < input > output
```

### Constract ASR dataset

Here, you can use [whisper](https://github.com/openai/whisper) to constract <speech, ASR-transcript> pair.

```python
import whisper

model = whisper.load_model("base.en")
result = model.transcribe("audio.mp3")
print(result["text"])
```

And use prompt tag \<golden>/ \<asr> to indicate whether the transcript contains errors(you should add the  prompt tag to the dict). Below is the snapshot for the MuST-C en-de dataset.

```
id      audio   n_frames        tgt_text        src_text        speaker asr
ted_1_0 flac.zip:23645981165:26957      25920   Hinter mir war gar keine Autokolonne.   <golden> ▁There ▁was ▁no ▁motor c ade ▁back ▁there ▁.   spk.1   ▁There ▁was ▁no ▁motor c ade ▁back ▁there ▁.
ted_12352_26    flac.zip:18519928824:48920      43200   Aber sie selbst sind auch eine Art von Kunst.   <asr> ▁but ▁they ▁are ▁also ▁a ▁form ▁of ▁art ▁in ▁and ▁of ▁themselves ▁.    spk.12352       ▁But ▁they ▁' re ▁also ▁a ▁form ▁of ▁art ▁in ▁and ▁of ▁themselves ▁.
```



## Training and Inference

###  Pretrain the MT Module

Pretrain the MT model on `${PARALLEL_MT_DATA}`, and save checkpoint  on `${SAVE_MT_CHECKPOINT}`.

```
fairseq-train ${PARALLEL_MT_DATA} \
    --arch transformer_pretrained_mt_m --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 4096 --fp16 --eval-bleu --scoring sacrebleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --keep-last-epochs 10 --eval-bleu-remove-bpe \
    --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${SAVE_MT_CHECKPOINT} --log-format json --log-interval 100 \
    --patience 10 --skip-invalid-size-inputs-valid-test
```



### Training

```shell
bash train_en2x.sh de
```



### Evaluate

Evaluate our model on text, speech, fuse-speech-text.

```shell
bash test_en2x.sh de
```





## Citation











