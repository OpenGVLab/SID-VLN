import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import random

import sys
sys.path.append("..")
sys.path.append(".")


from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from utils.data import ImageFeaturesDB

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch, ImgNavBatch
from reverie.parser import parse_args

def build_dataset(args, rank=0):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_files, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    dataset_class = ReverieObjectNavBatch

    if args.aug is not None and int(args.aug_times) > 0:
        aug_env = ImgNavBatch(
            feat_db, args.aug, args.connectivity_dir, 
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, max_objects=args.max_objects,
            seed=args.seed+rank, sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='aug'
        )
    else:
        aug_env = None

    
    # args.aug_only = True
    if args.aug_only:
        train_env, aug_env = aug_env, None
        args.aug = None
    else:
        train_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['train'], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        train_env = dataset_class(
            feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='train', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )

    val_env_names = ['val_seen', 'val_unseen']#'val_train_seen', 
    aug_val_env_names = ['val_seen', 'val_unseen']#'val_train_seen',

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False,
        )   # evaluation using all objects
        val_envs[split] = val_env

    if int(args.aug_times) > 0:
        aug_val_envs = {}
        for split in aug_val_env_names:
            if split == 'val_seen':
                data_path = '../datasets/REVERIE/annotations/img_val_seen.json'
            else:
                data_path = '../datasets/REVERIE/annotations/img_val_unseen.json'
            aug_val_env = ImgNavBatch(
                feat_db, data_path, args.img_path, args.connectivity_dir,
                batch_size=args.batch_size,
                angle_feat_size=args.angle_feat_size, max_objects=args.max_objects, 
                seed=args.seed+rank, sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='aug'
            )
            aug_val_envs[split] = aug_val_env
    else:
        aug_val_envs = None

    return train_env, val_envs, aug_env, aug_val_envs


def train(args, train_env, val_envs, aug_env=None, aug_val_envs=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        aug_record_file = os.path.join(args.log_dir, 'aug_train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    else:
        writer,record_file,aug_record_file = None,None,None

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank)
    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    aug_best_val = {'val_unseen': {"spl": 0., "sr": 0., "similarity": -1.,"state":""}}
    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
    
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
        # return

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    if args.sel_data:# sample successful explored trajectories, modify as your style
        listner.load('your_path/ckpts/best_val_unseen')
        feat_db = ImageFeaturesDB(args.img_ft_files, args.image_feat_size)
        data_path=f'your_path/annotations/hm3d_pool/{args.scan}.jsonl' # split the shortest_path trajectories from HM3D based on scans
        print(data_path)
        aug_data_env = ImgNavBatch(
            feat_db, data_path, args.img_path, args.connectivity_dir,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, max_objects=args.max_objects, 
            seed=args.seed+rank, sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size),
            name='aug'
        )
        data_envs = {}
        data_envs['aug'] = aug_data_env
        extract_data(args, data_envs, listner, writer, 0, default_gpu, start, aug_best_val, aug_record_file, 1)
        exit()

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // (int(args.rvr_times)+int(args.aug_times))))
            for jdx in range(interval // (int(args.rvr_times)+int(args.aug_times))):
                # Train with Augmented data
                listner.env = aug_env
                listner.train(int(args.aug_times), feedback=args.feedback)

                # Train with GT data
                listner.env = train_env
                listner.train(int(args.rvr_times), feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                record_file
            )

        if int(args.rvr_times) > 0:
            valid_during_train(args, val_envs, listner, writer, idx, default_gpu, start, best_val, record_file, iter)
        # if int(args.aug_times) > 0:
        #     valid_during_train(args, aug_val_envs, listner, writer, idx, default_gpu, start, aug_best_val, aug_record_file, iter)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))
    
    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def valid_during_train(args, val_envs, listner, writer, idx, default_gpu, start, best_val, record_file, iter):
    # Run validation
    loss_str = "iter {}".format(iter)
    for env_name, env in val_envs.items():

        listner.env = env
        # Get validation distance from goal under test evaluation conditions
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        preds = listner.get_results()
        preds = merge_dist_results(all_gather(preds))
        if default_gpu:
            score_summary, _ = env.eval_metrics(preds)
            env.instr_id = {}
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

            # select model by spl + sr(Goal-Oriented)
            if env_name in best_val:
                if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                    best_val[env_name]['spl'] = score_summary['spl']
                    best_val[env_name]['sr'] = score_summary['sr']
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
    if default_gpu:
        # listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

        write_to_record_file(
            ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
            record_file
        )
        write_to_record_file("BEST RESULT TILL NOW", record_file)
        for env_name in best_val:
            write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)

def extract_data(args, val_envs, listner, writer, idx, default_gpu, start, best_val, record_file, iter):
    # Run validation
    loss_str = "iter {}".format(iter)
    for env_name, env in val_envs.items():
        listner.env = env
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        preds = listner.get_results()
        preds = merge_dist_results(all_gather(preds))
        if default_gpu:
            env.extract_data(preds)

def valid_viz(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))
        
    
    for env_name, env in val_envs.items():
        if env_name != 'val_unseen':
            continue

        prefix = 'viz'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters, viz=True)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))

        preds = []
        for k, v in agent.results.items():
            preds.append({'instr_id': k, 'path': v['path']})
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            json.dump(
                preds,
                open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def zero_shot(args, val_envs, rank):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'zero_shot.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    else:
        writer, record_file = None, None    
    agent_class = GMapObjectNavAgent
    ckpt_dir = "your_path/pretrain/duet_cap/ckpts"
    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    for file_name in os.listdir(ckpt_dir):
        args.bert_ckpt_file = os.path.join(ckpt_dir, file_name)
        listner = agent_class(args, None, rank=rank)
        loss_str = "iter {}".format(file_name)
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                env.instr_id = {}
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], int(file_name.split('_')[-1].replace('.pt', '')))
                
                
                # select model by spl + sr(Goal-Oriented)
                if env_name in best_val:
                    write_to_record_file(file_name, record_file)
                    write_to_record_file(str(score_summary), record_file)
                    if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'File %s %s' % (file_name, loss_str)
        if default_gpu:
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)

def main():
    args = parse_args()

    if args.world_size > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, aug_val_envs = build_dataset(args, rank=rank)

    # valid(args, train_env, val_envs, rank=rank)
    # zero_shot(args, val_envs, rank)
    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, aug_val_envs=aug_val_envs, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
        valid_viz(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
