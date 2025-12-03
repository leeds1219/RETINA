import os
import json
import re
import random
import requests
import argparse
from math import ceil
from collections import defaultdict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

def normalize_entity(e):
    e = e.lower()
    e = re.sub(r"[^\w\s]", "", e)
    e = re.sub(r"\s+", " ", e).strip()
    return e

def parse_qa(text: str):
    match = re.search(r'Question:\s*(.*?)\s*Answer:\s*(.*)', text, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer_block = match.group(2).strip()
        answer = answer_block.splitlines()[0].strip()
        return question, answer
    else:
        raise ValueError("Invalid QA format")

def get_wikipedia_image_url(title):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "titles": title,
        "pithumbsize": 1000
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id in pages:
        page = pages[page_id]
        if "thumbnail" in page:
            return page["thumbnail"]["source"]
    return None

def load_prompts(root_path):
    prompt_files = {
        "RELATED_ENTITY_SYSTEM_PROMPT": "RELATED_ENTITY_SYSTEM_PROMPT.txt",
        "RELATED_ENTITY_USER_PROMPT": "RELATED_ENTITY_USER_PROMPT.txt",
        "SOURCE_ENTITY_SYSTEM_PROMPT": "SOURCE_ENTITY_SYSTEM_PROMPT.txt",
        "SOURCE_ENTITY_USER_PROMPT": "SOURCE_ENTITY_USER_PROMPT.txt",
    }
    prompts = {}
    for key, filename in prompt_files.items():
        path = os.path.join(root_path, "Prompts", filename)
        with open(path, "r", encoding="utf-8") as f:
            prompts[key] = f.read()
    return prompts

def replace_entity_with_this_image(question_str, target_entity):
    pattern = re.compile(re.escape(target_entity), re.IGNORECASE)
    return pattern.sub("given image", question_str)

def is_low_quality_sample(sample):
    entity = sample.get("entity", "")
    relation = sample.get("relation", "")
    source_qa_q = sample.get("source_qa", {}).get("question", "").lower()
    if re.fullmatch(r"\d{4}", entity):
        return True
    if re.search(r"\d{4} was the \d{1,3}(st|nd|rd|th) year of the \d{1,2}(st|nd|rd|th) century", relation):
        return True
    generic_q_patterns = [
        r"what event did this year witness.*",
        r"which year of .* century is .*",
        r"in which century is this year.*",
    ]
    for pattern in generic_q_patterns:
        if re.fullmatch(pattern, source_qa_q):
            return True
    return False

def prepare_paired_prompts(entity_data, source_system_prompt, source_user_prompt, related_system_prompt, related_user_prompt, entity_lookup):
    paired_prompts = []
    for entry in tqdm(entity_data, desc="filtering"):
        if is_low_quality_sample(entry):
            continue
        source_entity = entry['entity']
        source_image_url = entry['image_url']
        source_titles = set(entry['titles'])
        related_entities = entry.get('related_entity', {})
        if not related_entities:
            continue
        for related_entity_name, related_entity_info in related_entities.items():
            related_titles = set(related_entity_info['titles'])
            if source_titles.intersection(related_titles):
                continue
            entity_info = entity_lookup.get(related_entity_name)
            if not entity_info:
                wiki_url = get_wikipedia_image_url(related_entity_name)
                if not wiki_url:
                    continue
            relation = related_entity_info['relation']
            source_title = related_entity_info.get('source_title', None)
            paired_prompts.append({
                "relation": relation,
                "entity": source_entity,
                "source_title": source_title,
                "related_entity": related_entity_name,
                "related_title": related_titles,
                "source_image_url": source_image_url,
                "source_prompt": {
                    "system": source_system_prompt,
                    "user": source_user_prompt.format(
                        relation=relation,
                        entity=source_entity,
                        related_entity=related_entity_name
                    )
                },
                "related_prompt": {
                    "system": related_system_prompt,
                    "user": related_user_prompt.format(
                        relation=relation,
                        entity=source_entity,
                        related_entity=related_entity_name
                    )
                }
            })
    return paired_prompts

def load_data_and_model(args, device):
    dataset_path = os.path.join(args.root_path, "multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR")
    llm_model_path = os.path.join(args.root_path, "Qwen2.5-32B-Instruct")

    entity_path1 = os.path.join(args.root_path, "entity_relation_graph", f"{args.dataset.lower()}_entity_relaton_graph.json")
    entity_path2 = os.path.join(args.root_path, "entity_relation_graph", f"{args.dataset.lower()}_entity_relation_graph_start_index_10000.json")

    with open(entity_path1, "r", encoding="utf-8") as f1:
        entity_data1 = [json.loads(line) for line in f1]
    with open(entity_path2, "r", encoding="utf-8") as f2:
        entity_data2 = [json.loads(line) for line in f2]

    entity_data = entity_data1 + entity_data2
    entity_lookup = {item['entity']: item for item in entity_data}

    passage_dataset = load_dataset(
        os.path.join(args.root_path, "multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"),
        f"{args.dataset}_passages"
    )
    passages_list = []
    passages_list.extend(passage_dataset["train_passages"])
    passages_list.extend(passage_dataset["test_passages"])
    if args.dataset.lower() != "infoseek":
        passages_list += passage_dataset["valid_passages"]
    if args.dataset.lower() == "infoseek":
        passage_dict = {normalize_entity(p["title"]): p for p in passages_list}
    else:
        passage_dict = {normalize_entity(p["passage_id"]): p for p in passages_list}

    model = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype="auto").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")

    return {
        "entity_data": entity_data,
        "entity_lookup": entity_lookup,
        "passage_dict": passage_dict,
        "model": model,
        "tokenizer": tokenizer
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--dataset", type=str, default="EVQA")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

def main():
    args = get_args()
    device = f"cuda:{args.device}"
    data_bundle = load_data_and_model(args, device)
    entity_data = data_bundle["entity_data"]
    entity_lookup = data_bundle["entity_lookup"]
    passage_dict = data_bundle["passage_dict"]
    model = data_bundle["model"]
    tokenizer = data_bundle["tokenizer"]

    prompts = load_prompts(args.root_path)
    SOURCE_SYSTEM_PROMPT = prompts["SOURCE_ENTITY_SYSTEM_PROMPT"]
    RELATED_SYSTEM_PROMPT = prompts["RELATED_ENTITY_SYSTEM_PROMPT"]
    SOURCE_USER_PROMPT = prompts["SOURCE_ENTITY_USER_PROMPT"]
    RELATED_USER_PROMPT = prompts["RELATED_ENTITY_USER_PROMPT"]

    paired_prompts = prepare_paired_prompts(
        entity_data,
        SOURCE_SYSTEM_PROMPT,
        SOURCE_USER_PROMPT,
        RELATED_SYSTEM_PROMPT,
        RELATED_USER_PROMPT,
        entity_lookup,
    )

    with open(args.output_file, 'r', encoding='utf-8') as f:
        previous_results = [json.loads(line) for line in f]

    existing_pairs = {
        (item.get("entity"), item.get("related_entity"))
        for item in previous_results
        if item.get("entity") and item.get("related_entity")
    }

    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        for batch_start in range(0, len(paired_prompts), args.batch_size):
            batch_indices = range(batch_start, min(batch_start + args.batch_size, len(paired_prompts)))

            filtered_indices = []
            for idx in batch_indices:
                pair = (paired_prompts[idx]["entity"], paired_prompts[idx]["related_entity"])
                if pair not in existing_pairs:
                    filtered_indices.append(idx)
            if not filtered_indices:
                continue

            flat_prompts = []
            meta_info = []
            for idx in filtered_indices:
                item = paired_prompts[idx]
                flat_prompts.append({
                    "system": item["source_prompt"]["system"],
                    "user": item["source_prompt"]["user"]
                })
                meta_info.append((idx, "source"))
                flat_prompts.append({
                    "system": item["related_prompt"]["system"],
                    "user": item["related_prompt"]["user"]
                })
                meta_info.append((idx, "related"))

            messages_list = [
                [
                    {"role": "system", "content": p["system"]},
                    {"role": "user", "content": p["user"]}
                ]
                for p in flat_prompts
            ]
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            input_lengths = [len(input_id) for input_id in model_inputs.input_ids]
            generated_ids = [
                output_ids[input_len:]
                for output_ids, input_len in zip(generated_ids, input_lengths)
            ]

            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for (orig_idx, qa_type), decoded in zip(meta_info, decoded_outputs):
                try:
                    q, a = parse_qa(decoded)
                except ValueError:
                    q, a = None, None
                if qa_type == "source":
                    paired_prompts[orig_idx]["source_qa"] = {"question": q, "answer": a}
                else:
                    paired_prompts[orig_idx]["related_qa"] = {"question": q, "answer": a}

            for idx in filtered_indices:
                output_dict = {
                    "relation": paired_prompts[idx]["relation"],
                    "entity": paired_prompts[idx]["entity"],
                    "related_entity": paired_prompts[idx]["related_entity"],
                    "source_qa": paired_prompts[idx].get("source_qa", {}),
                    "related_qa": paired_prompts[idx].get("related_qa", {}),
                    "source_title": paired_prompts[idx].get("source_title", {}),
                    "source_image_url": paired_prompts[idx]["source_image_url"],
                    "related_image_url": get_wikipedia_image_url(paired_prompts[idx]["related_entity"])
                }
                f_out.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
