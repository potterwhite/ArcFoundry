#!/bin/bash

set -e
V=1
if [[ "${V:0}" == 1 ]];then
	set -x
fi

N=2 # 每个说话人最多取2个
TOTAL_SAMPLES=250 # 最终想要的总数
SAVED_FILE="./calibration_candidates.txt"
DIRS=(
	"/development/v_docs/usar-samba-public/ai/data/train-clean/LibriSpeech/train-clean-100"	
)

if [ -f ${SAVED_FILE} ];then
	echo "${SAVED_FILE} exist, remove it first"
	rm -f "${SAVED_FILE}"
fi

for dir in "${DIRS[@]}";do
	echo -e "\ndir=${dir}"

	find ${dir} -type f -name "*.flac" -o -name "*.wav" | shuf | awk -F/ -v count=$N '{if (seen[$(NF-2)]++ < count) print}' | head -n $TOTAL_SAMPLES >> ${SAVED_FILE}
done

# ------------------------------------------------------------------------------------------
NEW_DIR="/development/v_docs/usar-samba-public/ai/data/aishell/data_aishell/wav/train"
find ${NEW_DIR} -type f -name "*.flac" -o -name "*.wav" | shuf | awk -F/ -v count=$N '{if (seen[$(NF-1)]++ < count) print}' | head -n $TOTAL_SAMPLES >> ${SAVED_FILE}

