# Set the path to save checkpoints
OUTPUT_DIR='output/prit_small_patch16_224'
# path to imagenet-1k set
DATA_PATH='~/Database/ILSVRC2017/Data/CLS-LOC/'
# path to pretrain model
MODEL_PATH='output/pretrain_prit_small_patch16_224/checkpoint-99.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_prit_class_finetuning.py \
        --model prit_small_patch16_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --update_freq 32\
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 50 \
        --warmup_epochs 5 \
        --dist_eval \
        --use_mean_pooling True \
        --log-wandb \
        --wandb-project 'prit-mae-finetune'
