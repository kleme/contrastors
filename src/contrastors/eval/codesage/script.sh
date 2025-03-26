#!/usr/bin/env bash

data_dir=/tmp/code_eval/nl2code
result_dir="results/codesage/"

export model_name_or_path=${1:-"nomic-uiuc/nomic-embed-code-v1"} 
export tokenizer_name_or_path=${1:-"nomic-ai/nomic-embed-text-v1-unsupervised"}
export prefixes=${1:-"search_query search_document"}
export dataset=${2:-"csn"}
export model_type=${2:-"nomic"}

# export model_name_or_path=${1:-"voyage"} 
# export tokenizer_name_or_path=${1:-""}
# export prefixes=${1:-""}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}

# export model_name_or_path=${1:-"nomic-ai/nomic-embed-text-v1"} 
# export tokenizer_name_or_path=${1:-"nomic-ai/nomic-embed-text-v1"}
# export prefixes=${1:-"search_query search_document"}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}

# export model_name_or_path=${1:-"Salesforce/codet5p-110m-embedding"} 
# export tokenizer_name_or_path=${1:-"Salesforce/codet5p-110m-embedding"}
# export prefixes=${1:-""}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}

# export model_name_or_path=${1:-"codesage/codesage-small"} 
# export tokenizer_name_or_path=${1:-"codesage/codesage-small"}
# export prefixes=${1:-""}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}

# export model_name_or_path=${1:-"jinaai/jina-embeddings-v2-base-code"} 
# export tokenizer_name_or_path=${1:-"jinaai/jina-embeddings-v2-base-code"}
# export prefixes=${1:-""}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}

# export model_name_or_path=${1:-"microsoft/unixcoder-base"} 
# export tokenizer_name_or_path=${1:-"microsoft/unixcoder-base"}
# export prefixes=${1:-""}
# export dataset=${2:-"csn"}
# export model_type=${2:-"transformers"}



export CUDA_VISIBLE_DEVICES=0

function advTest() {
    python3 -W ignore /workspace/contrastors-dev/src/contrastors/eval/codesage/nl2code_search.py \
        --dataset_name advtest \
        --model_name_or_path "$model_name_or_path" \
        --tokenizer_name_or_path "$tokenizer_name_or_path" \
        --result_dir $result_dir/advtest \
        --test_data_file $data_dir/AdvTest/test.jsonl \
        --codebase_file $data_dir/AdvTest/test.jsonl \
        --prefixes "$prefixes" \
        --model_type "$model_type" \
        --code_length 1024 \
        --nl_length 1024 \
        --eval_batch_size 256 \
        --seed 123456
}

function cosqa() {
    python3 -W ignore /workspace/contrastors-dev/src/contrastors/eval/codesage/nl2code_search.py \
        --dataset_name cosqa \
        --model_name_or_path "$model_name_or_path" \
        --tokenizer_name_or_path "$tokenizer_name_or_path" \
        --result_dir $result_dir/cosqa \
        --test_data_file $data_dir/cosqa/cosqa-retrieval-test-500.json \
        --codebase_file $data_dir/cosqa/code_idx_map.txt \
        --prefixes "$prefixes" \
        --model_type "$model_type" \
        --code_length 1024 \
        --nl_length 1024 \
        --eval_batch_size 128 \
        --seed 123456
}

function csn() {
    for lang in python java ruby php javascript go; do
        python3 -W ignore /workspace/contrastors-dev/src/contrastors/eval/codesage/nl2code_search.py \
            --dataset_name csn \
            --language $lang \
            --model_name_or_path "$model_name_or_path" \
            --tokenizer_name_or_path "$tokenizer_name_or_path" \
            --result_dir $result_dir/CSN \
            --test_data_file $data_dir/CSN/$lang/test.jsonl \
            --codebase_file $data_dir/CSN/$lang/codebase.jsonl \
            --prefixes "$prefixes" \
            --model_type "$model_type" \
            --code_length 1024 \
            --nl_length 1024 \
            --eval_batch_size 256 \
            --seed 123456
        wait
    done
}

cosqa
advTest
csn