#!/bin/bash
export CUDA_VISIBLE_DEVICES=

## the paths needs to be changed accordingly
processor_name=
checkpoint_path=

dataset=EVQA 
image_root_dir=
index_use_images="True"
title_key=passage_id # "title" for Infoseek
doc_image_root_dir=
doc_image_title2image=

run_indexing_setting="--run_indexing"
index_root_path="./indexes"

split=test
Ks="1 5 10 20 50 100"
sample_examples_setting="" # "--sample_examples 100000" for the train split on Infoseek

index_name=
experiment_name=
reports_dir="./reports" # dir to save the report

python preflmr_bulid_index.py \
  --use_gpu \
  ${run_indexing_setting} \
  --index_root_path "${index_root_path}" \
  --index_name "${index_name}" \
  --experiment_name "${experiment_name}" \
  --indexing_batch_size int \
  --image_root_dir "${image_root_dir}" \
  --dataset_hf_path "" \
  --dataset "${dataset}" \
  --use_split "${split}" \
  --nbits 8 \
  --num_gpus 1 \
  --Ks ${Ks} \
  --checkpoint_path "${checkpoint_path}" \
  --image_processor_name "${processor_name}" \
  --query_batch_size int \
  --save_report_path "${reports_dir}" \
  --doc_image_root_dir "${doc_image_root_dir}" \
  --doc_image_title2image "${doc_image_title2image}" \
  --index_use_images "${index_use_images}" \
  --title_key "${title_key}" \
  ${sample_examples_setting}
