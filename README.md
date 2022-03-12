# Introduction
This repository is used for double-blind submission. It can reproduce the main results on the task of NTU-60+ -> PKUMMD. We choose this task for instance since the datasets are simpler to access and process. This repository is based on [https://github.com/LinguoLi/CrosSCLR](https://github.com/LinguoLi/CrosSCLR).
# Install & Requirements
We conduct experiments on the following environment:
python == 3.7.4
pytorch == 1.7.1
CUDA == 10.1
Please refer to “requirements.txt” for detailed information.
# Data Preparation
First we download the NTU-RGB+D 120 dataset from [https://github.com/shahroudy/NTURGB-D](https://github.com/shahroudy/NTURGB-D) and
PKU Multi-Modality dataset from [https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html](https://github.com/shahroudy/NTURGB-D). Then we
follow the [https://github.com/LinguoLi/CrosSCLR](https://github.com/shahroudy/NTURGB-D) to preprocess the datasets and get the “.npy”
files. 
# Train and Evaluate
## Fill the 
