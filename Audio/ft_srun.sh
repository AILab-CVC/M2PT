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

exp_name=$3

batch_size=$4

audioset_train_json=/data/audioset/16k/train_cleaned.json
audioset_train_all_json=/data/audioset/16k/train_all.json
audioset_eval_json=/data/audioset/16k/eval_cleaned.json
audioset_label=/data/audioset/16k/class_labels_indices.csv
dataset=audioset


python -m torch.distributed.launch --nproc_per_node=8 main_finetune_as.py \
--log_dir ./as_exp/$exp_name \
--output_dir ./as_exp/$exp_name \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--roll_mag_aug True \
--epochs 60 \
--blr $blr \
--batch_size $batch_size \
--warmup_epochs 4 \
--first_eval_ep 15 \
--dist_eval \
--mask_2d True \
--mask_t_prob 0.2 \
--mask_f_prob 0.2 \