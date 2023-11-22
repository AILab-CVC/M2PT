pretrained_path=/Your/Path/Checkpoints/
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 16 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune ./mae_pretrain_vit_base.pth \
    --output_dir ./work_dir/pathway-linear_image \
    --log_dir ./work_dir/pathway-linear_image \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --pretrained_path $pretrained_path \