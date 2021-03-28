python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file fcos_Sna_VOC.yaml \
    DATALOADER.NUM_WORKERS 4 \
    SOLVER.IMS_PER_BATCH 16\
    TEST.IMS_PER_BATCH 16\
    OUTPUT_DIR training_dir/sn
