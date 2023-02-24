# CoDT
## Introduction
This repository is used for CoDT (Collaborating Domain-shared and Target-specific Feature Clustering for Cross-domain 3D Action Recognition), which is accepted by ECCV2022. It can reproduce the main results on the task of NTU-60+ -> PKUMMD. We choose this task as example because the datasets are easier to access and process. This repository is mainly based on [https://github.com/LinguoLi/CrosSCLR](https://github.com/LinguoLi/CrosSCLR). This is the initial version of the code, which has not been tested fully and will be further revised in the future.

We release the datasets that are used in our paper:

Link: https://rec.ustc.edu.cn/share/82720bc0-b33b-11ed-aa4d-332b7f14c602
Passward: ustc

## Install & Requirements
We conduct experiments on the following environment: 

* python == 3.7.4 
* pytorch == 1.7.1 
* CUDA == 10.1 

Please refer to `requirements.txt` for detailed information.
## Data Preparation
1. Download the NTU-RGB+D 120 dataset from [https://github.com/shahroudy/NTURGB-D](https://github.com/shahroudy/NTURGB-D) and
PKU Multi-Modality dataset from [https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html). 
2. Follow [https://github.com/LinguoLi/CrosSCLR](https://github.com/LinguoLi/CrosSCLR) to preprocess the datasets.
## Core Files
1. Configuration:
   
   `./config/ntu_to_pkummd.yaml`
2. Data loading: 
   
   `./feeder/feeder.py`
3. Model definition:

   `./net/model.py`
4. Training process: 

   `./processor/main.py`
## Train and Evaluate
1. Fill the `config/ntu_to_pkummd.yaml` with the stored paths of dataset. 
2. Train the model by run 
   ```
   bash main.sh
   ```
3. The models would be automatically evaluated every 5 epochs, and the log file and checkpoints would be saved in the folder `work_dir/ntu_to_mmd`.

## Citation
~~~~
@inproceedings{liu2022collaborating,
  title={Collaborating Domain-Shared and Target-Specific Feature Clustering for Cross-domain 3D Action Recognition},
  author={Liu, Qinying and Wang, Zilei},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={137--155},
  year={2022},
  organization={Springer}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Qinying Liu (lydyc@mail.ustc.edu.cn).
