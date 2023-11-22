## Multimodal Pathway for Audio Recognition


This repo builds based on the code and models of [Audio MAE](http://arxiv.org/abs/2207.06405).



### 1. Installation
- This repo follows the [MAE repo](https://github.com/facebookresearch/mae), Installation and preparation follow that repo.
- Copy files and patch the timm package by ``bash timm_patch.sh'' (Please change the path to your own timm package path).
- Please find [mae_env.yml](./mae_env.yml) for all the dependencies.
- You may also use download the conda-packed [conda env](https://drive.google.com/file/d/1ECVmVyscVqmhI7OQa0nghIsWVaZhZx3q/view?usp=sharing), untar it, and then:
```
conda env create -f mae_env.yaml
```

### 2. Prepare data:
Please download AudioSet at [here](https://research.google.com/audioset/). Due to copyright we cannot release the data. The data annotation json parased and used in this work is available [here](https://drive.google.com/file/d/1cAiaL69HFm1zSW4hqFQpdhNfHiVKBFNA/view?usp=share_link). The format follows the one in [AST](https://github.com/YuanGongND/ast). Please be sure to modify the path in the scripts accordingly to reflect your own setup.

### 3. Pretrianing on AudioSet-2M
For the brave ones to pre-train on AudioSet-2M: Please use the pretrain_audioset2M.sh by:
```
bash pretrain_audioset2M.sh
```
### 4. Fine-tuning on AudioSet-2M and AudioSet-20K

```
bash ft_srun_pathway.sh
```
