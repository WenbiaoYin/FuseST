#!/bin/bash

TGT_LANG=$1

PARAMS_EVAL="./data \
            --task speech_text_joint_to_text \
            --max-tokens 4000000 \
            --nbest 1 \
            --batch-size 128 \
            --config-yaml config.yaml \
            --scoring sacrebleu \
            --beam 10 --lenpen 1.0 \
            --max-source-positions 480000 \
            --max-tokens-text 2000 --max-positions-text 400 \
            --results-path ${RESULT_PATH} \
            --path ${SAVE_MODEL_DIR}/checkpoint_best.pt \
            --user-dir examples/speech_text_joint_to_text \
            --load-speech-only --skip-invalid-size-inputs-valid-test"

# test on ST data
echo "test  with checkpoint_best! "
for test_model_type in speech text text_speech; do # three distinct input modalities for translation
    for testset in tst-COMMON_asr tst-COMMON_golden; do
        fairseq-generate ${PARAMS_EVAL} --gen-subset $testset --test-model-type $test_model_type
        mv $result_best/generate-${testset}.txt $result_best/generate-${testset}-${test_model_type}.txt
    done
done

# test on mt data
fairseq-generate ${PARAMS_EVAL} \
    --gen-subset mt --test-model-type mt --load-text-valid-split test --max-tokens-text 20000 --max-positions-text 400 \
    --parallel-text-data ${PARALLEL_MT_DATA} --langpairs en-$TGT_LANG \
    --user-dir examples/speech_text_joint_to_text
