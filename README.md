# CoDT
## Introduction
This repository is used for double-blind submission. It can reproduce the main results on the task of NTU-60+ -> PKUMMD. We choose this task as example because the datasets are easier to access and process. This repository is mainly based on [https://github.com/LinguoLi/CrosSCLR](https://github.com/LinguoLi/CrosSCLR).
## Install & Requirements
We conduct experiments on the following environment: <br>
python == 3.7.4 <br>
pytorch == 1.7.1 <br>
CUDA == 10.1 <br>
Please refer to “requirements.txt” for detailed information.
## Core Files
1. Model definition:
   ./net/model.py
2. Training process: 
   ./processor/main.py
## Data Preparation
1. Download the NTU-RGB+D 120 dataset from [https://github.com/shahroudy/NTURGB-D](https://github.com/shahroudy/NTURGB-D) and
PKU Multi-Modality dataset from [https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html). 
2. Follow [https://github.com/LinguoLi/CrosSCLR](https://github.com/shahroudy/NTURGB-D) to preprocess the datasets.
## Train and Evaluate
1. Fill the “config/ntu_to_pkummd.yaml” with the stored paths of dataset. <br>
2. Train the model by run 
   ```
   bash main.sh
   ```
3. The models would be automatically evaluated every 5 epochs, and the log file and checkpoints would be saved in the folder “work_dir/ntu_to_mmd”.
