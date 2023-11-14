#!/bin/bash

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

inference_tag=decode_asr_asr_model_valid.acc.ave

log "$0 $*"

. utils/parse_options.sh

inference_expdir="$1/${inference_tag}"
for x in ${inference_expdir}/*; do
    if [ -d ${x} ]; then
        python local/score.py -m wer -i "${x}/score_wer/hyp.trn" -t "${x}/score_wer/ref.trn" -c -d -n -l single_space sub_words sub_patterns strip_hyphen lower -w nl_rm_fillers.lst nl_abbrev.lst -p nl_getallen100.lst nl_getallen1000.lst -r resources/ -o "${x}/score_wer"
        python local/score.py -m cer -i "${x}/score_cer/hyp.trn" -t "${x}/score_cer/ref.trn" -c -d -n -l single_space sub_words sub_patterns strip_hyphen lower -w nl_rm_fillers.lst nl_abbrev.lst -p nl_getallen100.lst nl_getallen1000.lst -r resources/ -o "${x}/score_cer"
    fi
done

echo "$0: Successfully wrote results"
