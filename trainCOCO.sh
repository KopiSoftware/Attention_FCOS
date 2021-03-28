python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file coco_psp_aug.yaml \
    DATALOADER.NUM_WORKERS 1 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    OUTPUT_DIR training_dir/fcos
