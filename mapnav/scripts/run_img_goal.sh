NODE_RANK=0
rand_num=$(($RANDOM % 1000 + 20000))
MASTER_PORT=$rand_num
DATA_ROOT=../datasets

train_alg=dagger

features=siglip_dinov2
ft_dim=1536
obj_features=vitbase
obj_ft_dim=768

ngpus=8
seed=0

outdir=${DATA_ROOT}/REVERIE/expr_duet/img_goal_ft

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 32
      --lr 2e-5
      --iters 200000
      --log_every 500
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      --rvr_times 1
      --aug_times 0
      
      --gamma 0."

# training with aug
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch \
    --nproc_per_node=${ngpus} --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --use_env reverie/main_nav_obj.py $flag  \
    --tokenizer bert \
    --bert_ckpt_file "your_path" \
    --aug "your_path/scale_round0_860scan.jsonl"