NODE_RANK=0
NUM_GPUS=8
outdir=../datasets/REVERIE/exprs_map/pretrain/img_pt

rand_num=$(($RANDOM % 1000 + 20000))
MASTER_PORT=$rand_num

# train
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --use_env \
    train.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/img_pt_config.json \
    --config config/img_pretrain.json \
    --output_dir $outdir