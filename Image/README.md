## Multimodal Pathway for Image Recognition

Our code is developed based on [MAE](https://arxiv.org/abs/2111.06377):


### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

```python
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```

### Finetuning

```bash
# Please modify this path first
pretrained_path=/Your/Path/Checkpoints/
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune mae_pretrain_vit_base.pth \
    --output_dir ./work_dir/pathway_image \
    --log_dir ./work_dir/pathway_image \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --pretrained_path $pretrained_path \
```
