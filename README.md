# Dual_Ent_LigUnity
Hyperbolic LigUnity with dual entailment

## Installation
To install all the dependencies, please run 
```
bash setup_env.sh
```
which installs the dependencies for LigUnity and the hyperbolic learning library.

Then download the Uni-Mol models via the command
```
git clone https://huggingface.co/dptech/Uni-Mol-Models
```

Then within the working directory, make a folder "save" and in it make a folder "train_log"

## Data Preparation
We can download the training data through the command 
```
wget --content-disposition "https://ndownloader.figshare.com/articles/27966819/versions/2"
```
inside the deisred data directory. Then unzip the ***.zip*** files in the downloaded folder, e.g. ***unzip TIME.zip***

## Experiments & Instructionss
To train the model, first change the ***data_path, finetune_mol_model, finetune_pocket_model*** attributes in the ***train.sh*** file to the approriate locations of the downloaded training dataset and Uni-Mol models.

Then the experiment can be simply run with 

```
bash train.sh
```

Please train two baselines:
1. simply run the code with train.sh
2. Go to unimol/models/unumol.py, and uncommented the line ***args.entailed = [1]*** in the ***build_model*** function, then run the code again