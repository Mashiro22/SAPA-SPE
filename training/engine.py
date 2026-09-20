import contextlib
import fcntl
import logging
import random
import os
import time
from os.path import join as j_

import numpy as np
import pandas as pd
import torch

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from mil_models import create_survival_model
from utils.losses import NLLSurvLoss, CoxLoss, MarginCappedCoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)



@contextlib.contextmanager
def cuda_eval_lock(context='eval'):
    """Serialize full-bag CUDA evaluation across parallel folds."""
    if not torch.cuda.is_available():
        yield
        return
    device_name = str(torch.cuda.current_device())
    lock_path = f"/tmp/otsurv_cuda_eval_{device_name}.lock"
    with open(lock_path, "w") as lock_file:
        logging.info(f"Waiting for CUDA eval lock for {context}: {lock_path}")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        logging.info(f"Acquired CUDA eval lock for {context}: {lock_path}")
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            logging.info(f"Released CUDA eval lock for {context}: {lock_path}")


def _early_stopper_state(stopper):
    if stopper is None:
        return None
    return {
        "best_score": stopper.best_score,
        "early_stop": stopper.early_stop,
        "counter": stopper.counter,
    }


def _load_early_stopper_state(stopper, state):
    if stopper is None or state is None:
        return
    stopper.best_score = state.get("best_score", stopper.best_score)
    stopper.early_stop = state.get("early_stop", stopper.early_stop)
    stopper.counter = state.get("counter", stopper.counter)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state):
    if not state:
        return
    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception as exc:
            logging.warning(f"Skip restoring python RNG state: {exc}")
    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception as exc:
            logging.warning(f"Skip restoring numpy RNG state: {exc}")
    if "torch" in state:
        try:
            torch_state = state["torch"]
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state.cpu().to(torch.uint8))
        except Exception as exc:
            logging.warning(f"Skip restoring torch RNG state: {exc}")
    if torch.cuda.is_available() and "cuda" in state:
        try:
            cuda_state = state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [s if torch.is_tensor(s) else torch.as_tensor(s, dtype=torch.uint8) for s in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:
            logging.warning(f"Skip restoring cuda RNG state: {exc}")


def save_latest_training_checkpoint(args, epoch, model, optimizer, lr_scheduler,
                                    early_stopper=None):
    if not getattr(args, "save_latest_checkpoint", 1):
        return
    path = j_(args.results_dir, "latest_checkpoint.pt")
    tmp_path = path + ".tmp"
    start = time.time()
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "early_stopper": _early_stopper_state(early_stopper),
        "rng_state": _rng_state(),
        "config": vars(args),
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    elapsed = time.time() - start
    size_mb = os.path.getsize(path) / (1024 ** 2)
    logging.info(
        f"Saved latest training checkpoint at epoch {epoch}: "
        f"{path} ({size_mb:.1f} MB, {elapsed:.2f}s)"
    )

def try_resume_latest_training_checkpoint(args, model, optimizer, lr_scheduler,
                                          early_stopper=None):
    if not getattr(args, "resume_latest_checkpoint", 1):
        return 0
    path = j_(args.results_dir, "latest_checkpoint.pt")
    if not os.path.isfile(path):
        return 0
    start = time.time()
    ckpt = torch.load(path, map_location=torch.device(args.device))
    model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    _load_early_stopper_state(early_stopper, ckpt.get("early_stopper"))
    _load_rng_state(ckpt.get("rng_state"))
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    logging.info(
        f"Resumed latest training checkpoint from {path}; "
        f"next epoch={start_epoch}, load_time={time.time() - start:.2f}s"
    )
    return start_epoch

def wait_for_cuda_free_fraction(min_free_fraction=0.60, sleep_seconds=30, context='eval'):
    if not torch.cuda.is_available():
        return
    try:
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception as exc:
        logging.warning(f"Skip GPU free-memory wait for {context}: {exc}")
        return

    threshold = int(total_bytes * min_free_fraction)
    while free_bytes < threshold:
        logging.warning(
            f"GPU free memory before {context} is {free_bytes / 1024**3:.2f} GiB / "
            f"{total_bytes / 1024**3:.2f} GiB; waiting {sleep_seconds}s for "
            f">= {min_free_fraction:.0%} free."
        )
        time.sleep(sleep_seconds)
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        threshold = int(total_bytes * min_free_fraction)
    logging.info(
        f"GPU free memory before {context}: {free_bytes / 1024**3:.2f} GiB / "
        f"{total_bytes / 1024**3:.2f} GiB."
    )




def _flatten_label_value(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().reshape(-1).tolist()
    return value


def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss' or args.es_metric == 'c_index'
    if args.early_stopping and 'val' not in datasets:
        raise ValueError('early_stopping=1 requires a validation split.')
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        eval_loss_fn = loss_fn
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        eval_loss_fn = loss_fn
    elif args.loss_fn == 'capped_cox':
        loss_fn = MarginCappedCoxLoss(margin=args.cox_margin)
        eval_loss_fn = loss_fn if getattr(args, 'capped_only_es', 0) else CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()
        eval_loss_fn = loss_fn
        capped_eval_loss_fn = None


    args.feat_dim = args.in_dim # Patch feature dimension
    logging.info('Init Model...')

    model = create_survival_model(args)
    model.to(torch.device(args.device))

    print_network(model)

    logging.info('Init optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        logging.info('Setup EarlyStopping...')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        logging.info('No EarlyStopping...')
        early_stopper = None

    start_epoch = try_resume_latest_training_checkpoint(
        args, model, optimizer, lr_scheduler, early_stopper)

    #####################
    # The training loop #
    #####################
    for epoch in range(start_epoch, args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        logging.info(f'{"#" * 10} TRAIN Epoch: {epoch} {"#" * 10}')
        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps, epoch=epoch)

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            if args.loss_fn == 'nll':
                val_tag = 'nll-loss'
            else:
                val_tag = 'capped-loss' if getattr(args, 'capped_only_es', 0) else 'cox-loss'
            logging.info(f'{"#" * 11} VAL Epoch: {epoch} {"#" * 11} [{val_tag}]')
            val_results, _ = validate_survival(model, datasets['val'], eval_loss_fn,
                                               print_every=args.print_every, verbose=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ### Check Early Stopping (Optional)
            if epoch > 10 and early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']
                elif args.es_metric == 'c_index':
                    score = val_results['c_index']
                else:
                    raise NotImplementedError
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        save_latest_training_checkpoint(
            args, epoch, model, optimizer, lr_scheduler,
            early_stopper)
        logging.info(f'{"#" * (22 + len(f"TRAIN Epoch: {epoch}"))}\n')

    ### End of epoch: load the selected checkpoint.
    selected_checkpoint = j_(args.results_dir, 's_checkpoint.pth')
    if args.early_stopping:
        model.load_state_dict(torch.load(selected_checkpoint, map_location=torch.device(args.device))['model'])
    else:
        # Keep one checkpoint schema for training and standalone inference.
        save_checkpoint(
            config=vars(args),
            epoch=max(start_epoch, args.max_epochs) - 1,
            model=model,
            score=None,
            save_dir=args.results_dir,
            fname='s_checkpoint.pth',
        )

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        logging.info(f'End of training. Evaluating on Split {k.upper()}...:')
        return_attn = args.model_type in ("otsurv_pg", "sapa_spe")
        results[k], dumps[k] = validate_survival(model, loader, eval_loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are

    
    latest_ckpt = j_(args.results_dir, "latest_checkpoint.pt")
    if os.path.isfile(latest_ckpt):
        os.remove(latest_ckpt)
        logging.info(f"Removed latest training checkpoint after successful final evaluation: {latest_ckpt}")
    return results, dumps


def test(datasets, args):
    """
    Test for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn in ('cox', 'capped_cox'):
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    logging.info('Init Model...')

    model = create_survival_model(args)
    model.to(torch.device(args.device))
    print_network(model)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    results, dumps = {}, {}
    for k, loader in datasets.items():
        logging.info(f'End of training. Evaluating on Split {k.upper()}...:')
        return_attn = args.model_type in ("otsurv_pg", "sapa_spe")
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        
    return results, dumps

## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32, epoch=0):
    
    model.train()
    if hasattr(model, 'reset_prompt_state'):
        model.reset_prompt_state()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    iterations_per_epoch = len(loader)
    iter_in_epoch = 0    
    for batch_idx, batch in enumerate(loader):
        device = next(model.parameters()).device
        if getattr(model, 'use_coords', False):
            data = [
                (
                    torch.Tensor(batch[i]['img']).to(device),
                    torch.Tensor(batch[i]['coords']).to(device),
                )
                for i in range(len(batch))
            ]
        else:
            data = [torch.Tensor(batch[i]['img']).to(device) for i in range(len(batch))]
        label = torch.Tensor([batch[i]['label'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        event_time = torch.Tensor([batch[i]['survival_time'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        censorship = torch.Tensor([batch[i]['censorship'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        iterations = epoch * iterations_per_epoch + iter_in_epoch
        data += [iterations_per_epoch]
        data += [iterations]
        out, log_dict = model(data, label=label, censorship=censorship, loss_fn=loss_fn)
        data = data[:-2]

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(np.mean([data[i][0].shape[0] if isinstance(data[i], tuple) else data[i].shape[0] for i in range(len(data))]), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            logging.info(msg)
        
        iter_in_epoch += 1

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    results['iterations_per_epoch'] = iterations_per_epoch
    results['iterations'] = iterations

    msg = [f"{k}: {v:.3f}" for k, v in results.items()]
    logging.info("\t".join(msg))

    del all_risk_scores, all_censorships, all_event_times
    torch.cuda.empty_cache()
    
    return results


@torch.no_grad()
def _validate_survival_unlocked(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,
                      train_results={'iterations_per_epoch':10, 'iterations':1000}):
    wait_for_cuda_free_fraction(context='validation/test')
    model.eval()
    if hasattr(model, 'reset_prompt_state'):
        model.reset_prompt_state()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_path_attn = []
    all_expert_logits = []
    all_pg_gate = []
    all_rpif_gate = []
    all_proto_weights = []
    all_proto_stats = []
    all_proto_stats_raw = []
    all_stat_score_scale = []

    for batch_idx, batch in enumerate(loader):
        device = next(model.parameters()).device
        if getattr(model, 'use_coords', False):
            data = [
                (
                    torch.Tensor(batch[i]['img']).to(device),
                    torch.Tensor(batch[i]['coords']).to(device),
                )
                for i in range(len(batch))
            ]
        else:
            data = [torch.Tensor(batch[i]['img']).to(device) for i in range(len(batch))]
        label = torch.Tensor([batch[i]['label'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        event_time = torch.Tensor([batch[i]['survival_time'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        censorship = torch.Tensor([batch[i]['censorship'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        data += [train_results['iterations_per_epoch']]
        data += [train_results['iterations']]
        out, log_dict = model(data, label=label, censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        data = data[:-2]

        if return_attn and 'path_attn' in out:
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        if 'expert_logits' in out:
            all_expert_logits.append(out['expert_logits'].detach().cpu().numpy())
        if 'pg_gate' in out:
            all_pg_gate.append(out['pg_gate'].detach().cpu().numpy())
        if 'rpif_gate' in out:
            all_rpif_gate.append(out['rpif_gate'].detach().cpu().numpy())
        if return_attn and 'proto_weights' in out:
            all_proto_weights.extend([x.detach().cpu().numpy() for x in out['proto_weights']])
        if return_attn and 'proto_stats' in out:
            all_proto_stats.extend([x.detach().cpu().numpy() for x in out['proto_stats']])
        if return_attn and 'proto_stats_raw' in out:
            all_proto_stats_raw.extend([x.detach().cpu().numpy() for x in out['proto_stats_raw']])
        if return_attn and 'stat_score_scale' in out:
            scale = out['stat_score_scale']
            if torch.is_tensor(scale):
                scale = scale.detach().cpu().numpy()
            all_stat_score_scale.append(np.asarray(scale))
        # End of iteration survival-specific metrics to calculate / log
        # bag_size_meter.update(data.size(1), n=len(data))
        bag_size_meter.update(np.mean([data[i][0].shape[0] if isinstance(data[i], tuple) else data[i].shape[0] for i in range(len(data))]), n=len(data))
    
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            logging.info(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn and len(all_path_attn) > 0:
        all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, (CoxLoss, MarginCappedCoxLoss)):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        logging.info("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn and len(all_path_attn) > 0:
            dumps['all_path_attn'] = all_path_attn
        if len(all_expert_logits) > 0:
            expert_logits = np.concatenate(all_expert_logits, axis=0)
            dumps['all_expert_logits'] = expert_logits
            dumps['all_expert_risks'] = np.exp(expert_logits)
        if len(all_pg_gate) > 0:
            dumps['pg_gate'] = np.concatenate(all_pg_gate, axis=0)
        if len(all_rpif_gate) > 0:
            dumps['rpif_gate'] = np.concatenate(all_rpif_gate, axis=0)
        if len(all_proto_weights) > 0:
            dumps['proto_weights'] = np.stack(all_proto_weights, axis=0)
        if len(all_proto_stats) > 0:
            dumps['proto_stats'] = np.stack(all_proto_stats, axis=0)
        if len(all_proto_stats_raw) > 0:
            dumps['proto_stats_raw'] = np.stack(all_proto_stats_raw, axis=0)
        if len(all_stat_score_scale) > 0:
            dumps['stat_score_scale'] = np.asarray(all_stat_score_scale)
    
    del all_risk_scores, all_censorships, all_event_times
    torch.cuda.empty_cache()

    return results, dumps


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,
                      train_results={'iterations_per_epoch':10, 'iterations':1000}):
    with cuda_eval_lock(context='validation/test'):
        return _validate_survival_unlocked(
            model, loader,
            loss_fn=loss_fn,
            print_every=print_every,
            dump_results=dump_results,
            recompute_loss_at_end=recompute_loss_at_end,
            return_attn=return_attn,
            verbose=verbose,
            train_results=train_results,
        )
