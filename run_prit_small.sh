# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_prit_mae_small_patch16_224'
# path to imagenet-1k train set
DATA_PATH='~/Database/ILSVRC2017/Data/CLS-LOC/train'


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_prit_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_prit_mae_small_patch16_224 \
        --batch_size 512 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --epochs 100 \
        --use_mean_pooling True \
        --output_dir ${OUTPUT_DIR} \
        --log-wandb \
        --wandb-project 'prit-mae'
