# Set the path to save checkpoints
OUTPUT_DIR='output/'
# path to imagenet-1k set
DATA_PATH='/home/xingyan/Database/ILSVRC2017/Data/CLS-LOC'
# path to pretrain model
MODEL_PATH='vit_small_pretrain.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --use_mean_pooling True \
    --log-wandb \
    --wandb-project 'mae-pytorch-finetune'