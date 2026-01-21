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

## Experiments & Instructionss

The experiment can be simply run with 

```
train.sh
```

Please train two baselines:
1. simply run the code with train.sh
2. Go to unimol/models/unumol.py, and uncommented the line ***args.entailed = [1]*** in the ***build_model*** function, then run the code again