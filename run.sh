#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

source /users/spraak/pwang/.bashrc
echo "Activate conda environment cu116"

export PATH=/usr/local/bin:$PATH

export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:$LD_LIBRARY_PATH

train_set=train_omk
valid_set=train_omk
test_sets=dev_omk

asr_tag=wat_omk
asr_config=conf/wat/dim256.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --asr_task asr_wat \
    --stage 2 \
    --stop_stage 13 \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang nl \
    --ngpu 1 \
    --nj 1 \
    --inference_nj 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --asr_stats_dir "asr_stats_wat_omk" \
    --inference_config "${inference_config}" \
    --gpu_inference true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
