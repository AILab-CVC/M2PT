#!/bin/bash
#SBATCH --job-name=aud
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=240GB
#SBATCH --signal=USR1@120
#SBATCH --constraint=volta32gb

#SBATCH --output=/checkpoint/%u/jobs/%A.out
#SBATCH --error=/checkpoint/%u/jobs/%A.err

if [ -z "$1" ]
then
	blr=1e-3
else
	blr=$1
fi

if [ -z "$2" ]
then
	ckpt=/checkpoint/berniehuang/experiments/53415548/checkpoint-20.pth
else
	ckpt=$2
fi


audioset_train_json=/data/audioset/16k/train_cleaned.json
audioset_train_all_json=/data/audioset/16k/train_all.json
audioset_eval_json=/data/audioset/16k/eval_cleaned.json
audioset_label=/data/audioset/16k/class_labels_indices.csv
dataset=audioset


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_as.py \
--log_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \
--output_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--roll_mag_aug True \
--epochs 60 \
--blr $blr \
--batch_size 8 \
--warmup_epochs 4 \
--first_eval_ep 15 \
--dist_eval \
--mask_2d True \
--mask_t_prob 0.2 \
--mask_f_prob 0.2 \
