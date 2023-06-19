#!/usr/bin/env python

import argparse
import json
import os
import random
import uuid
from datetime import datetime as dt
from time import time
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data.distributed
from torch import distributed as dist
from torch import nn, optim
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import (DataLoader, DistributedSampler, RandomSampler,
                              SequentialSampler)

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS,
                         RunningMetric, format_seconds, is_main_process,
                         setup_multi_processes, setup_slurm, validate)


def main_worker(config: Dict[str, Any], args: argparse.Namespace):
    if not args.distributed:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        args.rank = 0
        args.world_size = 1
    else:
        # initializes the distributed backend which will take care of synchronizing nodes/GPUs
        setup_multi_processes(config)
        setup_slurm("nccl", port=args.master_port)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Start running DDP on: {args.rank}.")
        config["training"]["batch_size"] = int(
            config["training"]["batch_size"] / args.world_size
        )
        # create model and move it to GPU with id rank
        device = args.rank
        torch.cuda.set_device(device)
        if is_main_process():
            print("BatchSize per GPU: ", config["training"]["batch_size"])
        dist.barrier()

    ##############################
    ########## DATASET ###########
    ##############################
    # Datasets loading
    is_normals = config["model"]["output_dim"] > 1
    save_dir = os.path.join(args.base_path, config["data"]["data_root"])
    assert hasattr(
        custom_dataset, config["data"]["train_dataset"]
    ), f"{config['data']['train_dataset']} not a custom dataset"
    train_dataset = getattr(custom_dataset, config["data"]["train_dataset"])(
        test_mode=False,
        base_path=save_dir,
        crop=config["data"]["crop"],
        augmentations_db=config["data"]["augmentations"],
    )
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=save_dir, crop=config["data"]["crop"]
    )

    if is_normals:
        metrics_tracker = RunningMetric(list(DICT_METRICS_NORMALS.keys()))
    else:
        metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    # Dataset samplers, create distributed sampler pinned to rank
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    else:
        print("\t-> Local random sampler")
        train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    # Dataset loader
    val_batch_size = 2 * config["training"]["batch_size"]
    num_workers = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    ##############################
    ########### MODEL ############
    ##############################
    # Build model
    model = IDisc.build(config).to(device)
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )

    ##############################
    ######### OPTIMIZER ##########
    ##############################
    f16 = config["training"].get("f16", False)
    nsteps_accumulation_gradient = config["training"]["nsteps_accumulation_gradient"]
    gen_model = model.module if args.distributed else model
    params, lrs = gen_model.get_params(config)
    optimizer = optim.AdamW(
        params,
        weight_decay=config["training"]["wd"],
    )

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lrs,
        total_steps=config["training"]["n_iters"],
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=-1,
        pct_start=0.3,
        div_factor=config["training"]["div_factor"],
        final_div_factor=config["training"]["final_div_factor"],
    )

    scaler = torch.cuda.amp.GradScaler()
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)
    optimizer.zero_grad()

    ##############################
    ########## TRAINING ##########
    ##############################
    true_step, step = 0, 0
    start = time()
    n_steps = config["training"]["n_iters"]
    validate.best_loss = np.inf

    if is_main_process():
        print("Start training:")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"
    while True:
        for batch in train_loader:
            if (step + 1) % nsteps_accumulation_gradient:
                with context as fp, model.no_sync() as no_sync:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    preds, losses, _ = model(**batch)
                    loss = (
                        sum([v for k, v in losses["opt"].items()])
                        / nsteps_accumulation_gradient
                    )
                if f16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient accumulation (if any), now sync gpus
            else:
                with context:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    preds, losses, _ = model(**batch)
                    loss = (
                        sum([v for k, v in losses["opt"].items()])
                        / nsteps_accumulation_gradient
                    )
                if f16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            step += 1
            true_step = step // nsteps_accumulation_gradient

            if is_main_process():
                # Train loss logging
                if step % (100 * nsteps_accumulation_gradient) == 0:
                    log_loss_dict = {
                        f"Train/{k}": v.detach().cpu().item()
                        for loss in losses.values()
                        for k, v in loss.items()
                    }
                    elapsed = int(time() - start)
                    eta = int(elapsed * (n_steps - true_step) / max(1, true_step))
                    print(
                        f"Loss at {true_step}/{n_steps} [{format_seconds(elapsed)}<{format_seconds(eta)}]:"
                    )
                    print(
                        ", ".join([f"{k}: {v:.5f}" for k, v in log_loss_dict.items()])
                    )

            #  Validation
            is_last_step = true_step == config["training"]["n_iters"]
            is_validation = (
                step
                % (
                    nsteps_accumulation_gradient
                    * config["training"]["validation_interval"]
                )
                == 0
            )
            if is_last_step or is_validation:
                del preds, losses, loss, batch
                torch.cuda.empty_cache()

                if is_main_process():
                    print(f"Validation at {true_step}th step...")
                model.eval()
                start_validation = time()
                with torch.no_grad():
                    validate(
                        model,
                        test_loader=valid_loader,
                        step=true_step,
                        config=config,
                        run_id=run_id,
                        save_dir=save_dir,
                        metrics_tracker=metrics_tracker,
                        context=context,
                    )
                if is_main_process():
                    print(f"Elapsed: {format_seconds(int(time() - start_validation))}")
                model.train()

        # Exit
        if true_step == config["training"]["n_iters"]:
            dist.destroy_process_group()
            return 0


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description="Training script", conflict_handler="resolve"
    )

    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--master-port", type=str, required=False)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--base-path", default=os.environ.get("TMPDIR", ""))

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # fix seeds
    seed = config["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main_worker(config, args)
