#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from easydict import EasyDict
from sklearn.metrics import f1_score
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from cmt_module import CMTConfig
from dvtie_dataset import DVTIEDataset, dvtie_collator
from modeling_dvtie import DVTIENet, DVTIENetConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)
logging.getLogger().handlers = []


def to_easydict(value):
    if isinstance(value, dict):
        return EasyDict({k: to_easydict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_easydict(v) for v in value]
    return value


def load_yaml_config(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def flatten_scalar_config(prefix: str, value, out: Dict[str, object]):
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            flatten_scalar_config(next_prefix, v, out)
        return
    if isinstance(value, (bool, int, float, str)):
        out[prefix] = value


def parse_tags(tags: str) -> List[str]:
    if not tags:
        return []
    return [t for t in tags.split("#") if t]


def resolve_mode(cli_mode: str, cfg: Dict) -> str:
    if cli_mode != "auto":
        return cli_mode
    if "mode" in cfg and cfg["mode"] in {"train", "infer", "train_infer"}:
        return cfg["mode"]
    if bool(cfg.get("eval_only", False)):
        return "infer"
    return "train_infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DVTIE training/inference pipeline")
    parser.add_argument("--config", "--config_file", dest="config_file", default="config/train.yaml")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "train", "infer", "train_infer"],
        help="auto: infer from eval_only in yaml",
    )
    parser.add_argument("--run_name", default="", help="Optional run folder name")
    parser.add_argument("--output_dir", default="", help="Override yaml output_dir")
    parser.add_argument(
        "--pretrained_model_path",
        default="",
        help="Override yaml pretrained_model_path",
    )
    return parser.parse_args()


def build_dataset_for_split(args: EasyDict, set_name: str, split: str) -> DVTIEDataset:
    anno_file = os.path.join(args.annotation_root, f"{set_name}_mask3d_{split}.json")
    if not os.path.isfile(anno_file):
        anno_file = os.path.join(args.annotation_root, f"{set_name}_{split}.json")
    if not os.path.isfile(anno_file):
        raise FileNotFoundError(f"Missing annotation file: {anno_file}")

    attn_maps_file = os.path.join(args.attn_maps_root, f"infer_attn_maps_{split}_{set_name}.pt")
    if not os.path.isfile(attn_maps_file):
        raise FileNotFoundError(f"Missing attention map file: {attn_maps_file}")

    attributes_file = os.path.join(args.annotation_root, f"scannet_mask3d_{split}_attributes.pt")
    if not os.path.isfile(attributes_file):
        raise FileNotFoundError(f"Missing attributes file: {attributes_file}")

    dataset_kwargs = {}
    for field in ["feat_file", "img_feat_file", "max_obj_num", "feat_dim", "img_feat_dim"]:
        if hasattr(args, field):
            dataset_kwargs[field] = getattr(args, field)

    return DVTIEDataset(
        anno_file=anno_file,
        attn_maps_file=attn_maps_file,
        attibutes_file=attributes_file,
        use_ori_attn_maps=bool(args.use_ori_attn_maps),
        use_mentioned_oids_in_answers=bool(args.use_mentioned_oids_in_answers),
        **dataset_kwargs,
    )


def setup_dataloaders(args: EasyDict, need_train: bool) -> Tuple[Optional[DataLoader], Dict[str, DataLoader]]:
    train_loader = None
    if need_train:
        train_sets = parse_tags(getattr(args, "train_tags", ""))
        if not train_sets:
            raise ValueError("train mode requires non-empty train_tags")
        train_datasets = [build_dataset_for_split(args, name, "train") for name in train_sets]
        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.train_batch_size),
            shuffle=True,
            collate_fn=dvtie_collator,
            num_workers=int(args.num_workers),
        )

    val_sets = parse_tags(getattr(args, "val_tags", ""))
    if not val_sets:
        raise ValueError("val_tags cannot be empty")

    val_loaders = {}
    for set_name in val_sets:
        dataset = build_dataset_for_split(args, set_name, "val")
        val_loaders[set_name] = DataLoader(
            dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            collate_fn=dvtie_collator,
            num_workers=int(args.num_workers),
        )

    return train_loader, val_loaders


def get_num_params(model: torch.nn.Module) -> int:
    params = filter(lambda p: p.requires_grad, model.parameters())
    return int(sum(np.prod(p.size()) for p in params))


def get_optimizer(model: DVTIENet, args: EasyDict) -> Optimizer:
    optimizer_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = float(args.text_lr) if "text_encoder" in name else float(args.lr)
        weight_decay = 0.0 if len(param.shape) == 1 or name.endswith(".bias") else float(args.weight_decay)
        optimizer_params.append({"params": param, "lr": lr, "weight_decay": weight_decay})
    return optim.AdamW(optimizer_params, betas=(0.9, 0.999))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_multi: float = 0.0,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr_multi, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_multi, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, args: EasyDict):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        min_lr_multi=float(args.min_lr_multi),
    )


def is_distributed(accelerator: Accelerator) -> bool:
    return bool(accelerator.use_distributed and accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized())


def merge_loss_dict_across_ranks(loss_dict: Dict[str, List[float]], accelerator: Accelerator) -> Dict[str, List[float]]:
    if not is_distributed(accelerator):
        return loss_dict
    gathered = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, loss_dict)
    merged = {}
    for rank_obj in gathered:
        for k, v in rank_obj.items():
            merged.setdefault(k, [])
            merged[k].extend(v)
    return merged


def merge_attn_map_dict_across_ranks(attn_maps: Dict[int, Dict[str, torch.Tensor]], accelerator: Accelerator):
    if not is_distributed(accelerator):
        return attn_maps
    gathered = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, attn_maps)
    merged = {}
    for rank_obj in gathered:
        merged.update(rank_obj)
    return merged


def load_model_checkpoint(
    model: torch.nn.Module,
    accelerator: Accelerator,
    ckpt_path: str,
    strict: bool = True,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
):
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    unwrapped = accelerator.unwrap_model(model)
    msg = unwrapped.load_state_dict(state_dict, strict=strict)
    if accelerator.is_main_process:
        logger.info(f"Loaded checkpoint: {ckpt_path}")
        logger.info(str(msg))

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LambdaLR],
    accelerator: Accelerator,
    ckpt_path: Path,
    step: int,
    epoch: int,
):
    if not accelerator.is_main_process:
        return
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "epoch": int(epoch),
        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, ckpt_path)


@torch.no_grad()
def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    save_path: Optional[Path] = None,
    timing_cfg: Optional[EasyDict] = None,
):
    model.eval()
    total_loss = {}
    attn_maps = {}

    timing_enable = False
    timing_warmup = 0
    if timing_cfg is not None:
        timing_enable = bool(getattr(timing_cfg, "enable", False))
        timing_warmup = int(getattr(timing_cfg, "warmup_iters", 0))

    timing_total_ms = 0.0
    timing_bert_ms = 0.0
    timing_cmt_ms = 0.0
    timing_total_samples = 0

    device = next(model.parameters()).device

    for step, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not accelerator.is_local_main_process,
    ):
        if timing_enable:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

        outputs = model(**batch)

        if timing_enable:
            if device.type == "cuda":
                end_event.record()
                end_event.synchronize()
                batch_total_ms = start_event.elapsed_time(end_event)
            else:
                batch_total_ms = (time.perf_counter() - start_time) * 1000.0

            batch_size = outputs["logits"].shape[0] if torch.is_tensor(outputs.get("logits")) else 0
            if step >= timing_warmup and batch_size > 0:
                timing_total_ms += batch_total_ms
                timing_total_samples += batch_size
                model_timing = outputs.get("timing", {})
                timing_bert_ms += float(model_timing.get("bert_ms", 0.0))
                timing_cmt_ms += float(model_timing.get("cmt_ms", 0.0))

        for key, value in outputs.items():
            if "loss" in key:
                total_loss.setdefault(key, [])
                total_loss[key].append(float(value.detach().float().cpu().item()))

        logits = outputs["logits"]
        gt_attn = batch["attn_maps"]
        num_obj = logits.shape[-1]

        for k in [5, 10, 20]:
            topk = min(k, num_obj)
            pred = torch.zeros_like(logits)
            pred_topk_idx = torch.topk(logits, k=topk, dim=-1).indices
            pred.scatter_(1, pred_topk_idx, 1)

            target = torch.zeros_like(gt_attn)
            target_topk_idx = torch.topk(gt_attn, k=topk, dim=-1).indices
            target.scatter_(1, target_topk_idx, 1)

            pred_np = pred.cpu().int().numpy()
            target_np = target.cpu().int().numpy()

            metric_key = f"topk_{k}_f1_score"
            total_loss.setdefault(metric_key, [])
            for i in range(target_np.shape[0]):
                f1 = f1_score(
                    target_np[i],
                    pred_np[i],
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                )[1]
                total_loss[metric_key].append(float(f1))

        if save_path is not None:
            for idx, attn_map in zip(batch["index"], logits):
                attn_maps[int(idx)] = {"a_map": attn_map.detach().float().cpu()}

    total_loss = merge_loss_dict_across_ranks(total_loss, accelerator)

    if save_path is not None:
        attn_maps = merge_attn_map_dict_across_ranks(attn_maps, accelerator)
        if accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_maps, save_path)

    metrics = {}
    for key, values in total_loss.items():
        if values:
            metrics[key] = float(np.mean(values))

    if timing_enable:
        total_ms_t = torch.tensor(timing_total_ms, device=device, dtype=torch.float64)
        bert_ms_t = torch.tensor(timing_bert_ms, device=device, dtype=torch.float64)
        cmt_ms_t = torch.tensor(timing_cmt_ms, device=device, dtype=torch.float64)
        total_samples_t = torch.tensor(timing_total_samples, device=device, dtype=torch.float64)

        if is_distributed(accelerator):
            dist.all_reduce(total_ms_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(bert_ms_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cmt_ms_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)

        if total_samples_t.item() > 0:
            metrics["timing_total_ms"] = float(total_ms_t.item() / total_samples_t.item())
            metrics["timing_bert_ms"] = float(bert_ms_t.item() / total_samples_t.item())
            metrics["timing_cmt_ms"] = float(cmt_ms_t.item() / total_samples_t.item())

    model.train()
    return metrics


def evaluate_named_loaders(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    split_prefix: str,
    accelerator: Accelerator,
    output_dir: Path,
    save_attn_maps: bool,
    timing_cfg: Optional[EasyDict],
):
    all_metrics = {}
    for set_name, loader in loaders.items():
        save_path = None
        if save_attn_maps:
            save_path = output_dir / f"infer_attn_maps_{split_prefix}_{set_name}.pt"
        metrics = evaluate_dataloader(
            model=model,
            dataloader=loader,
            accelerator=accelerator,
            save_path=save_path,
            timing_cfg=timing_cfg,
        )
        if accelerator.is_main_process:
            logger.info(f"[{split_prefix}] {set_name} metrics: {json.dumps(metrics, indent=2)}")
        for key, value in metrics.items():
            all_metrics[f"{split_prefix}_{set_name}_{key}"] = float(value)
    return all_metrics


def choose_metric(metrics: Dict[str, float], args: EasyDict):
    metric_key = getattr(args, "save_metric_key", "val_scanrefer_topk_5_f1_score")
    larger_better = bool(getattr(args, "save_metric_larger_better", True))

    if metric_key in metrics:
        return metric_key, float(metrics[metric_key]), larger_better

    for key in sorted(metrics.keys()):
        if "loss" in key:
            return key, float(metrics[key]), False

    if metrics:
        first_key = sorted(metrics.keys())[0]
        return first_key, float(metrics[first_key]), True

    return "", 0.0, True


def init_run_dir(args: EasyDict, accelerator: Accelerator, mode: str, run_name_override: str) -> Path:
    run_dir_str = None
    if accelerator.is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_prefix = "eval_dvtie" if mode == "infer" else "train_dvtie"
        run_name = run_name_override.strip() if run_name_override else str(getattr(args, "run_name", "")).strip()
        if not run_name:
            run_name = f"{default_prefix}-{timestamp}"
        run_dir_str = str(Path(args.output_dir) / run_name)

    shared = [run_dir_str]
    if is_distributed(accelerator):
        dist.broadcast_object_list(shared, src=0)
    run_dir = Path(shared[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_defaults(cfg: Dict):
    defaults = {
        "output_dir": "outputs",
        "seed": 42,
        "train_tags": "",
        "val_tags": "",
        "use_ori_attn_maps": False,
        "use_mentioned_oids_in_answers": True,
        "use_spatial_dec": True,
        "train_text_encoder": False,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "num_workers": 4,
        "gradient_accumulation_steps": 1,
        "max_epoch": 1,
        "warmup_epochs": 0.1,
        "val_interval": 1.0,
        "logging_steps": 10,
        "eval_train_set": False,
        "text_lr": 8e-5,
        "lr": 8e-4,
        "weight_decay": 0.01,
        "min_lr_multi": 0.001,
        "max_grad_norm": 5.0,
        "save_attn_maps": True,
        "pretrained_model_path": "",
        "mode": "train_infer",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)


def main():
    cli_args = parse_args()

    cfg_dict = load_yaml_config(cli_args.config_file)
    ensure_defaults(cfg_dict)

    if cli_args.output_dir:
        cfg_dict["output_dir"] = cli_args.output_dir
    if cli_args.pretrained_model_path:
        cfg_dict["pretrained_model_path"] = cli_args.pretrained_model_path

    mode = resolve_mode(cli_args.mode, cfg_dict)
    cfg_dict["mode"] = mode
    cfg_dict["eval_only"] = mode == "infer"

    args = to_easydict(cfg_dict)

    set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    run_dir = init_run_dir(args, accelerator, mode, cli_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        tracker_cfg = {}
        flatten_scalar_config("", cfg_dict, tracker_cfg)
        accelerator.init_trackers(project_name=run_dir.name, config=tracker_cfg)

        with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=False)
        with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    accelerator.wait_for_everyone()

    timing_cfg = to_easydict(getattr(args, "timing", {}))
    model_config = DVTIENetConfig(
        mm_encoder=CMTConfig(spatial_dec=bool(args.use_spatial_dec)),
        train_text_encoder=bool(args.train_text_encoder),
        enable_timing=bool(getattr(timing_cfg, "enable", False)),
    )
    if hasattr(args, "roberta_path"):
        model_config.roberta_path = str(args.roberta_path)
    if hasattr(args, "max_obj_num"):
        model_config.max_obj_num = int(args.max_obj_num)

    model = DVTIENet(model_config).train()
    trainable_params_m = get_num_params(model) / 1e6

    need_train = mode in {"train", "train_infer"}
    need_infer = mode in {"infer", "train_infer"}

    train_loader, val_loaders = setup_dataloaders(args, need_train=need_train)

    optimizer = get_optimizer(model, args) if need_train else None

    if need_train:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model = accelerator.prepare(model)

    for key in list(val_loaders.keys()):
        val_loaders[key] = accelerator.prepare(val_loaders[key])

    pretrained_raw = getattr(args, "pretrained_model_path", "")
    pretrained_path = "" if pretrained_raw is None else str(pretrained_raw).strip()

    if mode == "infer":
        if not pretrained_path or not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "infer mode requires a valid pretrained_model_path in yaml or --pretrained_model_path"
            )
        load_model_checkpoint(model, accelerator, pretrained_path, strict=True)
    elif pretrained_path and os.path.isfile(pretrained_path):
        load_model_checkpoint(model, accelerator, pretrained_path, strict=False)

    global_step = 0
    best_score = None
    best_ckpt_path = run_dir / "checkpoint_best.pth"
    last_ckpt_path = run_dir / "checkpoint_last.pth"

    if need_train:
        num_updates_per_epoch = math.ceil(len(train_loader) / int(args.gradient_accumulation_steps))
        max_train_steps = max(1, num_updates_per_epoch * int(args.max_epoch))
        max_train_epochs = math.ceil(max_train_steps / num_updates_per_epoch)
        eval_steps = (
            int(args.val_interval)
            if isinstance(args.val_interval, int)
            else int(float(args.val_interval) * num_updates_per_epoch)
        )
        eval_steps = max(1, eval_steps)

        lr_scheduler = get_scheduler(
            optimizer,
            num_warmup_steps=int(float(args.warmup_epochs) * num_updates_per_epoch),
            num_training_steps=max_train_steps,
            args=args,
        )

        logger.info("***** Running DVTIE training *****")
        logger.info(f"  Trainable parameters = {trainable_params_m:.3f}M")
        logger.info(f"  Num train examples = {len(train_loader) * int(args.train_batch_size)}")
        for name, loader in val_loaders.items():
            logger.info(f"  Num val examples ({name}) = {len(loader) * int(args.eval_batch_size)}")
        logger.info(f"  Num epochs = {max_train_epochs}")
        logger.info(f"  Eval every {eval_steps} optimizer steps")

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ncols=100,
        )

        rolling_losses = {}
        did_eval = False

        for epoch in range(max_train_epochs):
            set_seed(int(args.seed) + epoch)
            progress_bar.set_description(f"epoch: {epoch + 1}")

            for batch in train_loader:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        outputs = model(**batch)

                    loss = outputs["loss"]
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and float(args.max_grad_norm) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

                    optimizer.step()
                    optimizer.zero_grad()

                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()

                for key, value in outputs.items():
                    if "loss" in key:
                        reduced = accelerator.gather(value.detach().float()).mean().item()
                        rolling_losses[key] = rolling_losses.get(key, 0.0) + reduced

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logging_steps = max(1, int(args.logging_steps))
                    if global_step % logging_steps == 0:
                        log_payload = {"learning_rate": lr_scheduler.get_last_lr()[-1]}
                        for key, total in rolling_losses.items():
                            log_payload[f"train_{key}"] = total / logging_steps
                        accelerator.log(log_payload, step=global_step)
                        rolling_losses = {}

                    if global_step % eval_steps == 0:
                        did_eval = True
                        val_metrics = evaluate_named_loaders(
                            model=model,
                            loaders=val_loaders,
                            split_prefix="val",
                            accelerator=accelerator,
                            output_dir=run_dir,
                            save_attn_maps=False,
                            timing_cfg=timing_cfg,
                        )

                        all_metrics = dict(val_metrics)
                        if bool(args.eval_train_set):
                            train_metrics = evaluate_named_loaders(
                                model=model,
                                loaders={"train": train_loader},
                                split_prefix="train",
                                accelerator=accelerator,
                                output_dir=run_dir,
                                save_attn_maps=False,
                                timing_cfg=timing_cfg,
                            )
                            all_metrics.update(train_metrics)

                        if all_metrics:
                            accelerator.log(all_metrics, step=global_step)

                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            accelerator=accelerator,
                            ckpt_path=last_ckpt_path,
                            step=global_step,
                            epoch=epoch,
                        )

                        metric_name, metric_value, larger_better = choose_metric(all_metrics, args)
                        should_save_best = False
                        if metric_name:
                            if best_score is None:
                                should_save_best = True
                            elif larger_better and metric_value > best_score:
                                should_save_best = True
                            elif (not larger_better) and metric_value < best_score:
                                should_save_best = True

                        if should_save_best:
                            best_score = metric_value
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                accelerator=accelerator,
                                ckpt_path=best_ckpt_path,
                                step=global_step,
                                epoch=epoch,
                            )

                    if global_step >= max_train_steps:
                        break

            if global_step >= max_train_steps:
                break

        if not did_eval:
            val_metrics = evaluate_named_loaders(
                model=model,
                loaders=val_loaders,
                split_prefix="val",
                accelerator=accelerator,
                output_dir=run_dir,
                save_attn_maps=False,
                timing_cfg=timing_cfg,
            )
            if val_metrics:
                accelerator.log(val_metrics, step=global_step)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            accelerator=accelerator,
            ckpt_path=last_ckpt_path,
            step=global_step,
            epoch=max(0, int(args.max_epoch) - 1),
        )

    if need_infer:
        if mode == "infer":
            infer_ckpt = pretrained_path
        else:
            infer_ckpt = str(best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path)

        if not infer_ckpt or not os.path.isfile(infer_ckpt):
            raise FileNotFoundError(f"Cannot find checkpoint for inference: {infer_ckpt}")

        load_model_checkpoint(model, accelerator, infer_ckpt, strict=True)

        final_metrics = evaluate_named_loaders(
            model=model,
            loaders=val_loaders,
            split_prefix="val",
            accelerator=accelerator,
            output_dir=run_dir,
            save_attn_maps=bool(args.save_attn_maps),
            timing_cfg=timing_cfg,
        )

        if final_metrics:
            final_log_payload = {f"final_{k}": v for k, v in final_metrics.items()}
            accelerator.log(final_log_payload, step=max(1, global_step))

    if accelerator.is_main_process:
        logger.info(f"Run directory: {run_dir}")
        try:
            tracker = accelerator.get_tracker("tensorboard")
            tracker.finish()
        except Exception:
            pass

    accelerator.end_training()


if __name__ == "__main__":
    main()
