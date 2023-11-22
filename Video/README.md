# Multimodal Pathway for Video Recognition

This part of code is based on [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2). Thanks for their outstanding project.
## Usage

### 1. Environment Setup.

```
pip install -r requirements.txt
```

###  2. Prepare Data. 
*You can easily prepare data with OpenData Lab by running commands below*
```
pip install openxlab
openxlab dataset get --dataset-repo OpenMMLab/Kinetics-400
```

### 3. Train and evaluate model. 
We provide the experiment scripts for easier use:

* Kinetics-400 Dataset
```
bash run.sh
```
*Please edit `run.sh` before running the code.*
