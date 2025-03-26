# Text Scripts for Code Data Curation

## Downloading CodeSearchNet Data

To download the data: 
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="code-search-net/code_search_net",
    repo_type="dataset",
)
```

To unzip all the files:
```bash
cd ~/.cache/huggingface/hub/datasets--code-search-net--code_search_net/snapshots/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data
!for file in *.zip; do unzip "$file" -d "${file%.*}"; done
```

