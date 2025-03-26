# Towards Improved Code Retrieval and Reranking



## Installation


### Contrastors

The `contrastors` library relies on custom kernels from the [Flash Attention](https://github.com/Dao-AILab/flash-attention) repository. To setup your enviornment you will need to follow the steps below.

Make sure that you have Cuda 11.8+. You can check this by running `nvcc --version` or if you already have torch installed you can run `python -c "import torch; print(torch.version.cuda)"`

Create a python venv and activate it

```bash
python3 -m venv env
source env/bin/activate
```

Install [torch](https://pytorch.org/get-started/locally/). See the torch docs for specific instructions for your system (e.g. the default CUDA torch supports is 12.1 as of 12/12/2023).

```bash
pip3 install torch torchvision torchaudio
```

Install wheel, packaging, ninja for Flash Attention (so the builds don't take too long)

```bash
pip install wheel packaging ninja setuptools
```

Install Flash Attention and the custom kernels

```bash
pip install --no-cache-dir flash-attn --no-build-isolation git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/fused_dense_lib git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy
```

Install the rest of the requirements and the package

```bash
pip install -e . 
```



### Code-RAG-Bench
Install code-rag-bench dependencies. 
```bash
cd code-rag-bench/
pip install -r requirements.txt
```
#### BM25
```sh
# install pyserini
pip install pyserini==0.25.0
# install openjdk-11 and maven (if you don't have any)
conda install -c conda-forge openjdk=11 maven -y
```
For more information of installing pyserini, please refer to [installation guide for pyserini](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

## Data Loading and Pre-processing 
### Loading the Pre-Processed Data 
To load the shards used for fine-tuning nomic-embed-code-v1, run
```bash
python scripts/code/load_transfer_hf.py --mode "load_from_hf"
```
### Running Pre-Processing From Scratch


## Code Embedding Model Training and Fine-Tuning 

### Code Embedding Model Finetuning

To reproduce the contrastive fine-tuning used to develop nomic-embed-code-v1, run

```bash
cd src/contrastors
torchrun --nproc-per-node=1 train.py --config=configs/train/contrastive_code_finetune.yaml --dtype=bf16
```
To use different models, data, or hyperparameters, update the config, `configs/train/contrastive_code_finetune.yaml`.

## Code RAG Evaluation 
First move into the `code-rag-bench/` folder.
```
cd code-rag-bench/
```

### Retrieval 
```
cd retrieval/
```
#### Dataset Preprocessing
Before running retrieval on a dataset, you need to create the datastore for it. Following
```
python create/${data_name}.py
# currently evaluated benchmarks for ${data_name}
# basic programming: 'humaneval', 'mbpp', 'csn', 'cosqa', 'advtest'
# repository-level: 'swebench_repo'
```
Certain datasets (humaneval and mbpp) have the queries leaked as part of the documents, making the retrieval very trivial. As a result, to convert these datasets into more challenging benchmarks for retrieval, we filter out the query from the documents. 

Run the following command after creating the datastore.
```
python postprocess/${data_name}.py
# supported datasets for ${data_name}
# basic programming: 'humaneval', 'mbpp'
```

#### Evaluating Retrieval
TODO



### Generation, Chunking, Re-Ranking, Etc.
 Generation, Chunking, Re-Ranking, Etc. are not tested yet. Refer to the README.md inside `code-rag-bench/` for instructions. 





