"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import json
import logging
import os
import sys
from os.path import join as j_

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.file_utils import save_pkl
from utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time, extract_patching_info, setup_logging)
from wsi_datasets import WSI_OTSurv_Dataset

from .engine import train


def collate_fn_list(batch):
    return batch  # return list directly, no padding


def audit_case_level_splits(csv_splits):
    """Fail fast if train cases leak into a held-out split.

    The official OTSurv folds use the same held-out cases for validation and
    test, so only train-versus-held-out overlap is prohibited here.
    """
    if 'train' not in csv_splits:
        return
    train_cases = set(csv_splits['train']['histo']['case_id'].astype(str))
    overlap_counts = {}
    for split in ('val', 'test'):
        if split not in csv_splits:
            continue
        heldout_cases = set(csv_splits[split]['histo']['case_id'].astype(str))
        overlap = train_cases.intersection(heldout_cases)
        overlap_counts[split] = len(overlap)
        if overlap:
            raise ValueError(
                f'Case-level leakage: {len(overlap)} train cases also occur in '
                f'{split}. Split by case_id, not only by slide_id.'
            )
    if 'val' in csv_splits and 'test' in csv_splits:
        val_cases = set(csv_splits['val']['histo']['case_id'].astype(str))
        test_cases = set(csv_splits['test']['histo']['case_id'].astype(str))
        logging.info(
            'case-level split audit: train-val=%d train-test=%d val-test=%d',
            overlap_counts.get('val', 0),
            overlap_counts.get('test', 0),
            len(val_cases.intersection(test_cases)),
        )


def save_risk_details_csv(results_dir, split, dumps):
    if 'all_expert_logits' not in dumps:
        return
    expert_logits = dumps['all_expert_logits']
    expert_risks = dumps.get('all_expert_risks')
    if expert_risks is None:
        expert_risks = torch.exp(torch.tensor(expert_logits)).numpy()
    pd.DataFrame({
        'sample_id': dumps['sample_ids'],
        'event_time': dumps['all_event_times'].reshape(-1),
        'censorship': dumps['all_censorships'].reshape(-1),
        'event_observed': 1 - dumps['all_censorships'].reshape(-1),
        'risk_final': dumps['all_risk_scores'].reshape(-1),
        'logit_base': expert_logits[:, 0],
        'logit_pg': expert_logits[:, 1],
        'logit_rpif': expert_logits[:, 2],
        'risk_base': expert_risks[:, 0],
        'risk_pg': expert_risks[:, 1],
        'risk_rpif': expert_risks[:, 2],
        'pg_gate': dumps.get('pg_gate', 0).reshape(-1),
        'rpif_gate': dumps.get('rpif_gate', 0).reshape(-1),
    }).to_csv(j_(results_dir, f'{split}_risk_details.csv'), index=False)


def save_proto_reliability_csv(results_dir, split, dumps):
    if 'proto_stats' not in dumps:
        return
    proto_stats = dumps['proto_stats']
    proto_stats_raw = dumps.get('proto_stats_raw')
    proto_weights = dumps.get('proto_weights')
    rows = []
    sample_ids = dumps['sample_ids']
    risk_scores = dumps['all_risk_scores'].reshape(-1)
    event_times = dumps['all_event_times'].reshape(-1)
    censorships = dumps['all_censorships'].reshape(-1)
    for sample_idx, sample_id in enumerate(sample_ids):
        for proto_idx in range(proto_stats.shape[1]):
            row = {
                'sample_id': sample_id,
                'sample_index': sample_idx,
                'prototype_index': proto_idx,
                'risk_final': risk_scores[sample_idx],
                'event_time': event_times[sample_idx],
                'censorship': censorships[sample_idx],
                'event_observed': 1 - censorships[sample_idx],
                'mass': proto_stats[sample_idx, proto_idx, 0],
                'log_mass': proto_stats[sample_idx, proto_idx, 1],
                'entropy': proto_stats[sample_idx, proto_idx, 2],
                'proto_std': proto_stats[sample_idx, proto_idx, 3],
            }
            if proto_stats_raw is not None:
                row.update({
                    'mass_raw': proto_stats_raw[sample_idx, proto_idx, 0],
                    'log_mass_raw': proto_stats_raw[sample_idx, proto_idx, 1],
                    'entropy_raw': proto_stats_raw[sample_idx, proto_idx, 2],
                    'proto_std_raw': proto_stats_raw[sample_idx, proto_idx, 3],
                })
            if proto_weights is not None:
                row['proto_weight'] = proto_weights[sample_idx, 0, proto_idx]
            rows.append(row)
    pd.DataFrame(rows).to_csv(j_(results_dir, f'{split}_proto_reliability.csv'), index=False)


def save_proto_attention_csv(results_dir, split, dumps):
    if 'proto_weights' not in dumps:
        return
    proto_weights = np.asarray(dumps['proto_weights']).squeeze(1)
    eps = 1e-8
    entropy = -(proto_weights * np.log(np.clip(proto_weights, eps, None))).sum(axis=1)
    max_weight = proto_weights.max(axis=1)
    pd.DataFrame({
        'sample_id': dumps['sample_ids'],
        'event_time': dumps['all_event_times'].reshape(-1),
        'censorship': dumps['all_censorships'].reshape(-1),
        'event_observed': 1 - dumps['all_censorships'].reshape(-1),
        'risk_final': dumps['all_risk_scores'].reshape(-1),
        'proto_attention_entropy': entropy,
        'proto_attention_norm_entropy': entropy / np.log(max(proto_weights.shape[1], 2)),
        'proto_attention_max_weight': max_weight,
        'proto_attention_min_weight': proto_weights.min(axis=1),
        'proto_attention_weight_std': proto_weights.std(axis=1),
        'proto_attention_effective_num': np.exp(entropy),
        'proto_attention_top_proto': proto_weights.argmax(axis=1),
        'proto_attention_top_weight': max_weight,
    }).to_csv(j_(results_dir, f'{split}_proto_attention.csv'), index=False)


def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    
    for k in csv_splits.keys():  # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSI_OTSurv_Dataset(df=df['histo'], **dataset_kwargs)

        # use custom collate_fn (enable when data is not of equal length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=dataset_kwargs.get('shuffle', False),
                                num_workers=num_workers,
                                collate_fn=collate_fn_list)  # key modification

        dataset_splits[k] = dataloader
        logging.info(f'split: {k}, n: {len(dataset)}')

        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()

    return dataset_splits

def main(args):
    # Setup logging
    log_file = j_(args.results_dir, 'training.log')
    setup_logging(log_file)
    
    # One public interface controls patch count for every split.
    # A non-positive value keeps the complete slide bag.
    num_patches = int(args.num_patches)
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=num_patches,
                        bag_sample_strategy=args.bag_sample_strategy,
                        shuffle=True
                        )

    # Keep patch-selection policy deterministic and consistent across
    # train/validation/held-out evaluation when a finite bag is used.
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=num_patches,
                      bag_sample_strategy=args.bag_sample_strategy,
                      shuffle=False
                      )

    all_results, all_dumps = {}, {}

    seed_torch(args.seed, args.device)
    csv_splits = read_splits(args)
    audit_case_level_splits(csv_splits)
    logging.info('successfully read splits for: ' + str(list(csv_splits.keys())))
    dataset_splits = build_datasets(csv_splits, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args)

    # Save results
    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
        save_risk_details_csv(args.results_dir, split, fold_dumps[split])
        save_proto_reliability_csv(args.results_dir, split, fold_dumps[split])
        save_proto_attention_csv(args.results_dir, split, fold_dumps[split])
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--in_dim', type=int, default=1024,
                    help='input patch-feature dimension')
parser.add_argument(
    '--num_patches', type=int, default=-1,
    help='patches per slide for train/validation/test; -1 uses the full bag',
)
parser.add_argument('--bag_sample_strategy', type=str, default='random',
                    choices=['random', 'spatial_uniform'],
                    help='patch subsampling strategy when a finite bag size is used')

### misc ###
parser.add_argument('--print_every', default=1,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=1, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=10,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

# model args ###
parser.add_argument(
    '--model_type', default='sapa_spe',
    choices=['otsurv', 'otsurv_pg', 'sapa_spe'],
    help='survival model included in this public release',
)
parser.add_argument('--spatial_embed_dim', type=int, default=16,
                    help='dimension of spatial footprint embedding')
parser.add_argument('--patch_encoder_type', type=str, default='original',
                    choices=['original', 'random_orthogonal'],
                    help='patch encoder; original reproduces the OTSurv/SAPA-SPE experiments')
parser.add_argument('--patch_encoder_seed', type=int, default=1,
                    help='seed for the optional fixed random orthogonal encoder')
parser.add_argument('--spatial_detach_stats', type=int, default=1,
                    help='detach spatial descriptors from OT assignments')
parser.add_argument('--spatial_use_extent', type=int, default=0,
                    help='append prototype extent descriptor')
parser.add_argument('--spatial_grid_size', type=int, default=8,
                    help='grid size for lightweight spatial descriptors')
parser.add_argument('--top_mass_ratio', type=float, default=0.70,
                    help='prototype attention mass used for top-region spatial descriptors')
parser.add_argument('--top_mass_max_nodes', type=int, default=1024,
                    help='maximum selected patches per prototype for spatial graph descriptors')
parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'capped_cox', 'rank'],
                    help='which loss function to use')
parser.add_argument('--cox_margin', type=float, default=1.0,
                    help='margin for one-sided capped Cox loss')
parser.add_argument('--capped_only_es', type=int, default=0,
                    help='for capped_cox, use capped loss for validation, early stopping, and final evaluation when set to 1')


parser.add_argument('--nll_alpha', type=float, default=0.5,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='BLCA_survival')
parser.add_argument('--target_col', type=str, default='dss_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source') # /your/data/path/feats_h5
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')  # e.g. src/splits/survival/TCGA_BLCA_overall_survival_k=0
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')
parser.add_argument('--save_latest_checkpoint', type=int, default=1,
                    help='save complete latest checkpoint after every completed epoch')
parser.add_argument('--resume_latest_checkpoint', type=int, default=1,
                    help='resume from latest_checkpoint.pt in results_dir when present')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')

# device args ###
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device to run the model on')

args = parser.parse_args()

if __name__ == "__main__":

    logging.info(f'task: {args.task}')
    args.split_dir = j_('splits', args.split_dir)
    logging.info(f'split_dir: {args.split_dir}')
    split_num = args.split_dir.split('/')[2].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        logging.info(f'data source: {src}')
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)
        mag, patch_size = extract_patching_info(os.path.dirname(src))
        if (mag < 0 or patch_size < 0):
            raise ValueError(f"invalid patching info parsed for {src}")
        check_params_same.append([feat_name, mag, patch_size])

        #### parse model name ####
        parsed = parse_model_name(feat_name) 
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})
    
    try:
        check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
        assert check_params_same.drop(['feats_name'],axis=1).drop_duplicates().shape[0] == 1
        logging.info("All data sources have the same feature extraction parameters.")
    except:
        logging.info("Data sources do not share the same feature extraction parameters. Exiting...")
        sys.exit()
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)
    
    ### setup results dir ###
    if args.exp_code is None:
        exp_code = f"{args.split_name_clean}::{args.model_type}::{feat_name}"
    else:
        pass
    
    result_base_dir = j_(args.results_dir,
                         args.task,
                         f'k={args.split_k}',
                         str(exp_code))
    resume_dir = None
    if getattr(args, 'resume_latest_checkpoint', 1):
        if os.path.isdir(result_base_dir):
            for name in sorted(os.listdir(result_base_dir), reverse=True):
                candidate = j_(result_base_dir, name)
                if not os.path.isdir(candidate):
                    continue
                if os.path.isfile(j_(candidate, 'summary.csv')):
                    continue
                if os.path.isfile(j_(candidate, 'latest_checkpoint.pt')):
                    resume_dir = candidate
                    break
    if resume_dir is not None:
        args.results_dir = resume_dir
        logging.info(f'Resuming unfinished result directory with latest checkpoint: {args.results_dir}')
    else:
        args.results_dir = j_(result_base_dir, f"Time::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    logging.info("\n################### Settings ###################")
    for key, val in vars(args).items():
        logging.info("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    logging.info("FINISHED!\n\n\n")
