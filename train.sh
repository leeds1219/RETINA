#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"

## the paths needs to be changed accordingly
checkpoint_path=
processor_name=

dataset=
sample_examples=-1 # -1 for EVQA
image_root_dir= # change accordingly
doc_use_images="True"
title_key="passage_id" # "passage_id" for EVQA
doc_image_root_dir= # change accordingly
doc_image_title2image= # change accordingly

num_train_epochs_setting="--num_train_epochs 1"

output_dir= # dir to save the model
run_name= # name to specify a run 


torchrun --nproc_per_node 1 train.py \
    --model_name_or_path "${checkpoint_path}" \
    --image_processor_name "${processor_name}" \
    --dataset_hf_path # add \
    --dataset "${dataset}" \
    --sample_examples "${sample_examples}" \
    --num_negative_examples 4 \
    --image_root_dir "${image_root_dir}" \
    --freeze_vision_encoder True \
    --freeze_text_encoder False \
    --mapping_structure_lr 1e-4 \
    --non_mapping_structure_lr 1e-5 \
    --doc_use_images "${doc_use_images}" \
    --doc_image_root_dir "${doc_image_root_dir}" \
    --doc_image_title2image "${doc_image_title2image}"\
    --output_dir "${output_dir}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    ${num_train_epochs_setting} \
    --do_eval False \
    --evaluation_strategy "no" \
    --eval_steps 20 \
    --save_strategy "epoch" \
    --save_only_model True \
    --save_total_limit 20 \
    --remove_unused_columns False \
    --seed 42 \
    --run_name "${run_name}" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --title_key passage_id