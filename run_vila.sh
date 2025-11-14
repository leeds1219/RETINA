#!/bin/bash
export CUDA_VISIBLE_DEVICES=

conv_mode="v1"
model_base=
model_path= # requires "lora" in the folder name

data_path=
images_base=

output_file=
chunks_dir=
mkdir -p "${chunks_dir}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
num_gpus=${#GPULIST[@]}
CHUNKS=${num_gpus}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
    python vila_model_vqa_loader.py \
        --model-path "${model_path}" \
        --model-base "${model_base}" \
        --question-file "${data_path}" \
        --image-folder "${images_base}" \
        --answers-file "${chunks_dir}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx $IDX \
        --temperature 0 \
        --max_new_tokens 512 \
        --conv-mode "${conv_mode}" &
done

wait

# Clear out the output file if it exists.

> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${chunks_dir}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
