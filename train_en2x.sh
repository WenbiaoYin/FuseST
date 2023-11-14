#!/usr/bin/env bash

TGT_LANG=$1
SAVE_MODEL_DIR=./checkpoint
SAVE_TENSORBOARD_DIR=./tensorboard_log

wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt

fairseq-train ./data \
    --arch fstnet_m \
    --criterion mutil_task_for_fuse_speech_text \
    --train-subset train --valid-subset dev \
    --parallel-text-data ${PARALLEL_MT_DATA} \
    --save-dir ${SAVE_MODEL_DIR} \
    --tensorboard-logdir ${SAVE_TENSORBOARD_DIR} \
    --speech-encoder-layers 6 --text-encoder-layers 6 --decoder-layers 6 \
    --load-pretrain-decoder ${SAVE_MT_CHECKPOINT}/checkpoint_best.pt \
    --load-pretrain-text-encoder-last ${SAVE_MT_CHECKPOINT}/checkpoint_best.pt \
    --load-pretrain-audio-encoder ./wav2vec_small.pt \
    --num-workers 8 --task speech_text_joint_to_text --update-mix-data \
    --user-dir examples/speech_text_joint_to_text --optimizer adam \
    --max-epoch 60 --lr 6e-5 --lr-scheduler inverse_sqrt \
    --update-freq 4 --clip-norm 10.0 --guide-alpha 0.8 \
    --disable-text-guide-update-num 10000 --label-smoothing 0.1 \
    --max-source-positions 480000 --max-tokens 600000 \
    --max-tokens-text 2000 --max-positions-text 400 \
    --seed 4 --dropout 0.15 --warmup-updates 20000 \
    --attentive-cost-regularization 0.02 --text-sample-ratio 0.25 \
    --text-input-cost-ratio 0.5 --log-format json \
    --langpairs en-${TGT_LANG} \
    --max-tokens-valid 2000000 --ddp-backend no_c10d --log-interval 100 \
    --config-yaml config.yaml --keep-last-epochs 10 --fp16 \
    --skip-invalid-size-inputs-valid-test --data-buffer-size 50 \
    --ctc-weight 1.0 --contrastive-weight 1.0
