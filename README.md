# Dataset
```
https://huggingface.co/datasets/Lee1219/RETINA
```
# Setup
```
# Navigate to a safe directory (like ~/projects)
mkdir -p ~/projects
cd ~/projects

# Clone each repository
git clone https://github.com/LinWeizheDragon/FLMR
git clone https://github.com/lhdeng-gh/MuKA
git clone https://github.com/NVlabs/VILA

# Navigate to a safe directory (like ~/projects)
mkdir -p ~/projects
cd ~/projects

# Clone each repository
git clone https://github.com/LinWeizheDragon/FLMR
git clone https://github.com/lhdeng-gh/MuKA
git clone https://github.com/NVlabs/VILA

# -----------------------------------------------
# Conda environment setup (recommended):
# - For FLMR (retriever): create a dedicated Conda environment inside the FLMR folder
#   Example:
#     cd FLMR
#     conda env create -f environment.yml   # or manually create
#
# - For VILA (generator): create a dedicated Conda environment inside the VILA folder
#   Example:
#     cd VILA
#     conda env create -f environment.yml   # or manually create
# -----------------------------------------------

# -----------------------------------------------
# File modifications needed (just note the paths, actual edits later):
# - FLMR/flmr/models/flmr/modeling_flmr.py          # modify to handle multi-image documents
# - FLMR/flmr/models/flmr/modeling_flmr_for_indexing.py  # modify for multi-image indexing
# - FLMR/third_party/ColBERT/colbert/modeling/utils.py
# - FLMR/third_party/ColBERT/colbert/tokenization/colbert.py
# - ... (other files as required)
# - VILA/llava/model/builder.py
# -----------------------------------------------
```
# Inference
```
./run.sh # to run the retriever
./run_vila.sh # to run the generator
```

# To-Do List
## **Evaluation**
- [x] BEM metric demo

## **Dataset**
- [ ] This should be prioritized — the current dataset is an unfiltered and unprocessed version.
- [ ] Clean-up path
- [ ] Remove unused columns
- [ ] Unify formatting
      
## **Guide**
- [ ] FLMR/flmr/models/flmr/modeling_flmr.py (to handle multi-image documents)
- [ ] FLMR/flmr/models/flmr/modeling_flmr_for_indexing.py (to handle multi-image documents during indexing)
- [ ] FLMR/third_party/ColBERT/colbert/modeling/utils.py 
- [ ] FLMR/third_party/ColBERT/colbert/tokenization/colbert.py
- [ ] ...
- [ ] VILA/llava/model/builder.py

# Acknowledgement

We adopt these codes to create this repository.

```
https://github.com/stanford-futuredata/ColBERT
https://github.com/LinWeizheDragon/FLMR
https://github.com/lhdeng-gh/MuKA
https://github.com/NVlabs/VILA
```
