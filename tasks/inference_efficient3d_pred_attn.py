# # # import datetime
# # # import logging
# # # import time
# # # from os.path import join
# # #
# # # import pandas as pd
# # # import torch
# # # import torch.distributed as dist
# # # import wandb
# # # from torch.utils.data import ConcatDataset
# # # from functools import partial
# # #
# # # from dataset import MetaLoader, create_dataset, create_loader, create_sampler
# # # from dataset.dataset_train import train_collate_fn
# # # from dataset.dataset_val import val_collate_fn
# # #
# # # from models.chat3d_fast_gt_attn import Chat3D
# # #
# # # from tasks.shared_utils import get_media_types, setup_model
# # # from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
# # # from utils.config_utils import setup_main
# # # from utils.distributed import get_rank, get_world_size, is_main_process
# # # from utils.logger import log_dict_to_wandb, setup_wandb
# # # from utils.eval import (
# # #     calc_scanrefer_score,
# # #     clean_answer,
# # #     calc_scan2cap_score,
# # #     calc_scanqa_score,
# # #     calc_sqa3d_score,
# # #     calc_multi3dref_score,
# # #     calc_referit3d_score,
# # #     calc_scanrefer_location_score,
# # #     calc_multi3dref_location_score,
# # # )
# # #
# # # from pycocoevalcap.bleu.bleu import Bleu
# # # from pycocoevalcap.meteor.meteor import Meteor
# # # from pycocoevalcap.rouge.rouge import Rouge
# # # from pycocoevalcap.cider.cider import Cider
# # #
# # # # from pycocoevalcap.spice.spice import Spice
# # # from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# # #
# # #
# # # import numpy as np
# # # from tqdm import tqdm
# # #
# # # import json
# # # import os
# # #
# # # logger = logging.getLogger(__name__)
# # # max_bleus = [0.0] * 4
# # #
# # # tokenizer = PTBTokenizer()
# # # scorers = [
# # #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
# # #     (Meteor(), "METEOR"),
# # #     (Rouge(), "ROUGE_L"),
# # #     (Cider(), "CIDEr"),
# # #     # (Spice(), "SPICE")
# # # ]
# # #
# # # max_global_step = 200000000
# # #
# # #
# # # def train(
# # #     model,
# # #     model_without_ddp,
# # #     train_loaders,
# # #     val_loaders,
# # #     optimizer,
# # #     epoch,
# # #     global_step,
# # #     device,
# # #     scheduler,
# # #     scaler,
# # #     config,
# # #     do_eval=True,
# # # ):
# # #     model.train()
# # #     model_without_ddp.llama_model.config.use_cache = False
# # #
# # #     metric_logger = MetricLogger(delimiter="  ")
# # #     eval_metric_logger = MetricLogger(delimiter="  ")
# # #     metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
# # #     loss_names = ["loss", "obj_norm", "obj_img_norm", "objid_norm", "scene_norm"]
# # #     media_types = get_media_types(train_loaders)
# # #
# # #     # tot_param = sum(p.numel() for p in model_without_ddp.parameters())
# # #     # trainable_param = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
# # #     # print(f"Total Params: {tot_param / 1e6}M")
# # #     # print(f"Trainable Params: {trainable_param / 1e6}M")
# # #     # exit()
# # #
# # #     for name in loss_names:
# # #         metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.6f}"))
# # #
# # #     header = f"Train Epoch: [{epoch}]"
# # #     log_freq = config.log_freq
# # #
# # #     if config.distributed:
# # #         for d in train_loaders:
# # #             d.sampler.set_epoch(epoch)
# # #     train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))
# # #
# # #     accum_iter = config.grad_accum_steps  # 1
# # #     print(f"TrainLoader Length: {len(train_loader)}")
# # #     eval_freq = len(train_loader)  # 2000
# # #
# # #     optimizer.zero_grad()
# # #     iterator = metric_logger.log_every(train_loader, log_freq, header)
# # #     for i, (media_type, batch) in enumerate(iterator):
# # #         for k in batch.keys():
# # #             if type(batch[k]) == torch.Tensor:
# # #                 batch[k] = batch[k].to(device)
# # #         loss_dict = model(**batch)
# # #         loss = loss_dict["loss"] / accum_iter
# # #
# # #         model.require_backward_grad_sync = (i + 1) % accum_iter == 0
# # #
# # #         scaler.scale(loss).backward()
# # #
# # #         if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
# # #             if config.optimizer.max_grad_norm > 0:
# # #                 scaler.unscale_(optimizer)
# # #                 torch.nn.utils.clip_grad_norm_(
# # #                     model.parameters(), config.optimizer.max_grad_norm
# # #                 )
# # #             scaler.step(optimizer)
# # #             optimizer.zero_grad()
# # #             scaler.update()
# # #         scheduler.step()
# # #
# # #         # logging
# # #         for name in loss_names:
# # #             if name not in loss_dict:
# # #                 continue
# # #             value = loss_dict[name]
# # #             value = value if isinstance(value, float) else value.item()
# # #             metric_logger.update(**{f"{name}": value})
# # #         metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
# # #
# # #         if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
# # #             logs = metric_logger.get_avg_dict()
# # #             log_dict_to_wandb(logs, step=global_step, prefix="train/")
# # #
# # #         global_step += 1
# # #
# # #         if do_eval and (
# # #             (i + 1) % eval_freq == 0
# # #             and (len(train_loader) - i >= eval_freq)
# # #             or i == len(train_loader) - 1
# # #         ):
# # #             val_metrics = evaluate_all(
# # #                 model, model_without_ddp, val_loaders, epoch, global_step, device, config
# # #             )
# # #             if is_main_process():
# # #                 for k, v in val_metrics.items():
# # #                     if k not in eval_metric_logger.meters:
# # #                         eval_metric_logger.add_meter(
# # #                             k, SmoothedValue(window=1, fmt="{value:.4f}")
# # #                         )
# # #                 eval_metric_logger.update(**val_metrics)
# # #             if is_main_process() and config.wandb.enable:
# # #                 logs = eval_metric_logger.get_avg_dict()
# # #                 log_dict_to_wandb(logs, step=global_step, prefix="val/")
# # #
# # #             if is_main_process():
# # #                 param_grad_dic = {
# # #                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
# # #                 }
# # #                 state_dict = model_without_ddp.state_dict()
# # #                 for k in list(state_dict.keys()):
# # #                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
# # #                         # delete parameters that do not require gradient
# # #                         del state_dict[k]
# # #                 save_obj = {
# # #                     "model": state_dict,
# # #                     # "optimizer": optimizer.state_dict(),
# # #                     # "scheduler": scheduler.state_dict(),
# # #                     # "scaler": scaler.state_dict(),
# # #                     "config": config,
# # #                     "epoch": epoch,
# # #                     "global_step": global_step,
# # #                 }
# # #                 if i != len(train_loader) - 1 and config.do_save and not config.debug:
# # #                     torch.save(
# # #                         save_obj,
# # #                         join(config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"),
# # #                     )
# # #         if global_step > max_global_step:
# # #             return global_step
# # #
# # #     # gather the stats from all processes
# # #     metric_logger.synchronize_between_processes()
# # #     logger.info(f"Averaged stats: {metric_logger.global_avg()}")
# # #     return global_step
# # #
# # #
# # # def evaluate_all(
# # #     model, model_without_ddp, val_loaders, epoch, global_step, device, config
# # # ):
# # #     logger.info("Start evaluating...")
# # #     model.eval()
# # #     # model_without_ddp.llama_model.config.use_cache = True
# # #     val_scores = {}
# # #     for val_loader in val_loaders:
# # #         new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config)
# # #         val_scores = {**val_scores, **new_val_scores}
# # #
# # #     logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
# # #     for k, v in val_scores.items():
# # #         logger.info(f"{k}: {v}")
# # #
# # #     if is_main_process() and getattr(config, "do_save", True):
# # #         # 仅保存需要梯度的参数（与 train() 一致）
# # #         param_grad_dic = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
# # #         state_dict = model_without_ddp.state_dict()
# # #         for k in list(state_dict.keys()):
# # #             if k in param_grad_dic.keys() and not param_grad_dic[k]:
# # #                 del state_dict[k]
# # #         save_obj = {
# # #             "model": state_dict,
# # #             "config": config,
# # #             "epoch": epoch,
# # #             "global_step": global_step,
# # #         }
# # #         # 文件名你可按需改，这里用 eval 标签
# # #         torch.save(save_obj, join(config.output_dir, f"ckpt_eval_{epoch:02d}_{global_step}.pth"))
# # #         logger.info(
# # #             f"[EVAL SAVE] Saved checkpoint to {join(config.output_dir, f'ckpt_eval_{epoch:02d}_{global_step}.pth')}")
# # #
# # #     model.train()
# # #     # model.module.llama_model.config.use_cache = False
# # #     return val_scores
# # #
# # #
# # # def evaluate(model, val_loader, epoch, global_step, device, config):
# # #     eval_name = val_loader.dataset.datasets[0].dataset_name
# # #     logger.info(f"Evaluating {eval_name}...")
# # #     if config.distributed:
# # #         val_loader.sampler.set_epoch(epoch)
# # #
# # #     sample_freq = len(val_loader) // 5 + 1
# # #     cosine_scores, l2_distances = [], []
# # #     save_preds = []
# # #     logger.info(f"batch-size={val_loader.batch_size} length(#batches)={len(val_loader)}")
# # #     for i, batch in tqdm(enumerate(val_loader)):
# # #         for k in batch.keys():
# # #             if type(batch[k]) == torch.Tensor:
# # #                 batch[k] = batch[k].to(device)
# # #         with torch.no_grad():
# # #             pred = model(**batch, is_eval=True)
# # #         # if "target_captions" in batch:
# # #         #     cosine_scores.append(pred["cosine_score"])
# # #         #     l2_distances.append(pred["l2_dis"])
# # #
# # #         if "custom_prompt" in batch:
# # #             # if len(batch["ref_captions"][0]) > 0:
# # #             #     target = batch["ref_captions"]
# # #             #     prompt = batch["custom_prompt"]
# # #             #     tmp_pred = [p.replace("\n", " ").strip() for p in pred]
# # #             #     tmp_target = ['\n'.join(p) for p in target]
# # #             #     if i % sample_freq == 0:
# # #             #         logger.info(f"\n[Prompt]\n{prompt[0]}\n[Pred]\n{tmp_pred[0]}\n[Target(s)]\n{tmp_target[0]}")
# # #             batch_size = len(pred)
# # #             for bi in range(batch_size):
# # #                 scene_id = batch["scene_id"][bi]
# # #                 obj_id = int(batch["obj_ids"][bi])
# # #                 qid = batch["qid"][bi]
# # #                 prompt = batch["custom_prompt"][bi]
# # #                 pred_id = int(batch["pred_ids"][bi])
# # #                 type_info = batch["type_infos"][bi]
# # #                 tmp_pred = pred[bi]
# # #                 save_preds.append(
# # #                     {
# # #                         "scene_id": scene_id,
# # #                         "gt_id": obj_id,
# # #                         "pred_id": pred_id,
# # #                         "qid": qid,
# # #                         "prompt": prompt,
# # #                         "pred": tmp_pred,
# # #                         "ref_captions": batch["ref_captions"][bi],
# # #                         "type_info": type_info,
# # #                     }
# # #                 )
# # #             # if i % sample_freq == 0:
# # #             #     print(save_preds[-1])
# # #
# # #     if len(save_preds) > 0:
# # #         save_preds = sorted(
# # #             save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
# # #         )
# # #         with open(
# # #             os.path.join(
# # #                 config.output_dir,
# # #                 f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json",
# # #             ),
# # #             "w",
# # #         ) as f:
# # #             json.dump(save_preds, f, indent=4)
# # #
# # #     dist.barrier()
# # #     if is_main_process():
# # #         save_preds = []
# # #         for rank in range(config.gpu_num):
# # #             path = os.path.join(
# # #                 config.output_dir,
# # #                 f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json",
# # #             )
# # #             if os.path.exists(path):
# # #                 preds = json.load(open(path, "r"))
# # #                 save_preds += preds
# # #                 os.remove(path)
# # #         save_preds = sorted(
# # #             save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
# # #         )
# # #         with open(
# # #             os.path.join(
# # #                 config.output_dir,
# # #                 f"preds_epoch{epoch}_step{global_step}_{eval_name}.json",
# # #             ),
# # #             "w",
# # #         ) as f:
# # #             json.dump(save_preds, f, indent=4)
# # #
# # #     val_scores = {}
# # #     if is_main_process() and len(save_preds) > 0:
# # #         if eval_name == "scanqa":
# # #             val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
# # #         elif eval_name == "scanrefer":
# # #             val_scores = calc_scanrefer_score(save_preds, config)
# # #         elif eval_name in ["scan2cap", "scan2cap_location"]:
# # #             val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
# # #         elif eval_name in ["sqa3d", "sqa3d_val"]:
# # #             val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config)
# # #         elif eval_name == "multi3dref":
# # #             val_scores = calc_multi3dref_score(save_preds, config)
# # #         elif eval_name in ["nr3d", "sr3d"]:
# # #             val_scores = calc_referit3d_score(save_preds, eval_name, config)
# # #         elif eval_name == "scanrefer_location":
# # #             val_scores = calc_scanrefer_location_score(save_preds, config)
# # #         elif eval_name == "multi3dref_location":
# # #             val_score = calc_multi3dref_location_score(save_preds, config)
# # #         elif eval_name in ["scanrefer_test", "scan2cap_test"]:
# # #             pass
# # #         print(json.dumps(val_scores, indent=4))
# # #     return val_scores
# # #
# # #
# # # def setup_dataloaders(config):
# # #     # train datasets, create a list of data loaders
# # #     train_datasets, val_datasets = create_dataset(config)
# # #
# # #     if config.distributed:
# # #         num_tasks = get_world_size()
# # #         global_rank = get_rank()
# # #         train_samplers = create_sampler(
# # #             train_datasets, [True] * len(train_datasets), num_tasks, global_rank
# # #         )
# # #         val_samplers = create_sampler(
# # #             val_datasets, [False] * len(val_datasets), num_tasks, global_rank
# # #         )
# # #     else:
# # #         train_samplers = [None] * len(train_datasets)
# # #         val_samplers = [None] * len(val_datasets)
# # #
# # #     train_loaders = create_loader(
# # #         train_datasets,
# # #         train_samplers,
# # #         batch_size=[config.batch_size] * len(val_datasets),
# # #         num_workers=[config.num_workers] * len(train_datasets),
# # #         is_trains=[True] * len(train_datasets),
# # #         collate_fns=[train_collate_fn] * len(train_datasets),
# # #     )
# # #     _val_collate_fn = partial(
# # #         val_collate_fn, use_external_attn_maps=config.use_external_attn_maps
# # #     )
# # #     val_loaders = create_loader(
# # #         val_datasets,
# # #         val_samplers,
# # #         batch_size=[config.batch_size] * len(val_datasets),
# # #         num_workers=[config.num_workers] * len(val_datasets),
# # #         is_trains=[False] * len(val_datasets),
# # #         collate_fns=[_val_collate_fn] * len(val_datasets),
# # #     )
# # #
# # #     return train_loaders, val_loaders
# # #
# # #
# # # def main(config):
# # #     if is_main_process() and config.wandb.enable:
# # #         run = setup_wandb(config)
# # #
# # #     # torch.autograd.set_detect_anomaly(True)
# # #     setup_seed(config.seed + get_rank())
# # #     device = torch.device(config.device)
# # #
# # #     train_loaders, val_loaders = setup_dataloaders(config)
# # #
# # #     num_steps_per_epoch = sum(len(d) for d in train_loaders)
# # #     config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
# # #     config.scheduler.num_warmup_steps = (
# # #         num_steps_per_epoch * config.scheduler.warmup_epochs
# # #     )
# # #     torch.backends.cudnn.benchmark = True
# # #
# # #     model_cls = eval(config.model.get("model_cls", "Chat3D"))
# # #     (
# # #         model,
# # #         model_without_ddp,
# # #         optimizer,
# # #         scheduler,
# # #         scaler,
# # #         start_epoch,
# # #         global_step,
# # #     ) = setup_model(
# # #         config,
# # #         model_cls=model_cls,
# # #         find_unused_parameters=True,
# # #     )
# # #     if is_main_process() and config.wandb.enable:
# # #         wandb.watch(model)
# # #
# # #     save_step_interval = 1
# # #     start_time = time.time()
# # #     if not config.evaluate:
# # #         logger.info("Start training")
# # #         for epoch in range(start_epoch, config.scheduler.epochs):
# # #             global_step = train(
# # #                 model,
# # #                 model_without_ddp,
# # #                 train_loaders,
# # #                 val_loaders,
# # #                 optimizer,
# # #                 epoch,
# # #                 global_step,
# # #                 device,
# # #                 scheduler,
# # #                 scaler,
# # #                 config,
# # #                 do_eval=config.do_eval,
# # #             )
# # #             if is_main_process():
# # #                 logger.info(f"Epoch {epoch}")
# # #                 param_grad_dic = {
# # #                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
# # #                 }
# # #                 state_dict = model_without_ddp.state_dict()
# # #                 for k in list(state_dict.keys()):
# # #                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
# # #                         # delete parameters that do not require gradient
# # #                         del state_dict[k]
# # #                 save_obj = {
# # #                     "model": state_dict,
# # #                     # "optimizer": optimizer.state_dict(),
# # #                     # "scheduler": scheduler.state_dict(),
# # #                     # "scaler": scaler.state_dict(),
# # #                     "config": config,
# # #                     "epoch": epoch,
# # #                     "global_step": global_step,
# # #                 }
# # #                 if (
# # #                     (
# # #                         (epoch + 1) % save_step_interval == 0
# # #                         or epoch == config.scheduler.epochs - 1
# # #                     )
# # #                     and config.do_save
# # #                     and not config.debug
# # #                 ):
# # #                     if config.get("save_latest", False):
# # #                         torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
# # #                     else:
# # #                         torch.save(
# # #                             save_obj,
# # #                             join(
# # #                                 config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"
# # #                             ),
# # #                         )
# # #
# # #             if global_step > max_global_step:
# # #                 break
# # #             dist.barrier()
# # #
# # #     if config.evaluate:
# # #         evaluate_all(
# # #             model,
# # #             model_without_ddp,
# # #             val_loaders,
# # #             start_epoch - 1,
# # #             global_step,
# # #             device,
# # #             config,
# # #         )
# # #
# # #     total_time = time.time() - start_time
# # #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
# # #     logger.info(f"Training time {total_time_str}")
# # #     logger.info(f"Checkpoints and Logs saved at {config.output_dir}")
# # #
# # #     if is_main_process() and config.wandb.enable:
# # #         run.finish()
# # #
# # #
# # # if __name__ == "__main__":
# # #     cfg = setup_main()
# # #     main(cfg)
# #
# #
# #
# # import datetime
# # import logging
# # import time
# # from os.path import join
# #
# # import pandas as pd
# # import torch
# # import torch.distributed as dist
# # import wandb
# # from torch.utils.data import ConcatDataset
# # from functools import partial
# #
# # from dataset import MetaLoader, create_dataset, create_loader, create_sampler
# # from dataset.dataset_train import train_collate_fn
# # from dataset.dataset_val import val_collate_fn
# #
# # from models.chat3d_fast_gt_attn import Chat3D
# #
# # from tasks.shared_utils import get_media_types, setup_model
# # from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
# # from utils.config_utils import setup_main
# # from utils.distributed import get_rank, get_world_size, is_main_process
# # from utils.logger import log_dict_to_wandb, setup_wandb
# # from utils.eval import (
# #     calc_scanrefer_score,
# #     clean_answer,
# #     calc_scan2cap_score,
# #     calc_scanqa_score,
# #     calc_sqa3d_score,
# #     calc_multi3dref_score,
# #     calc_referit3d_score,
# #     calc_scanrefer_location_score,
# #     calc_multi3dref_location_score,
# # )
# #
# # from pycocoevalcap.bleu.bleu import Bleu
# # from pycocoevalcap.meteor.meteor import Meteor
# # from pycocoevalcap.rouge.rouge import Rouge
# # from pycocoevalcap.cider.cider import Cider
# # from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# #
# # import numpy as np
# # from tqdm import tqdm
# #
# # import json
# # import os
# #
# # # ========== [新增] 运行时 FLOPs 计数器（MACs 口径 ≈ FLOPs） ==========
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from contextlib import contextmanager
# #
# # class _FlopsMeter:
# #     """
# #     推理期实时 FLOPs 计数器：
# #     - 统计 nn.Linear、nn.Conv2d 的 MACs
# #     - 包裹 PyTorch 2.x 的 fused scaled_dot_product_attention（SDPA）
# #     - 仅在 activate() 上下文内生效，不污染训练
# #     口径：MACs（乘加次数），行业常直接称为 FLOPs。若要严格 FLOPs≈2×MACs，可自行乘2。
# #     """
# #     def __init__(self):
# #         self.macs = 0
# #         self.handles = []
# #         self._orig_sdpa = None
# #
# #     def reset(self):
# #         self.macs = 0
# #
# #     def add(self, v: int):
# #         if v is not None:
# #             self.macs += int(v)
# #
# #     # 线性层：B*...*in*out
# #     def _linear_hook(self, mod: nn.Linear, inp, out):
# #         x = inp[0]
# #         if x is None:
# #             return
# #         in_features = mod.in_features
# #         out_features = mod.out_features
# #         elems = 1
# #         for d in x.shape[:-1]:
# #             elems *= int(d)
# #         self.add(elems * in_features * out_features)
# #
# #     # 卷积层：B * Cout * Hout * Wout * (Cin/groups * Kh * Kw)
# #     def _conv2d_hook(self, mod: nn.Conv2d, inp, out):
# #         x = inp[0]
# #         if x is None:
# #             return
# #         B = int(x.shape[0])
# #         Cout = int(out.shape[1])
# #         Hout = int(out.shape[2])
# #         Wout = int(out.shape[3])
# #         Cin = int(mod.in_channels)
# #         Kh, Kw = mod.kernel_size
# #         groups = int(mod.groups)
# #         macs_per_out = (Cin // groups) * Kh * Kw
# #         self.add(B * Cout * Hout * Wout * macs_per_out)
# #
# #     # 包裹 SDPA，依据 q/k/v 的真实形状统计注意力核心 FLOPs
# #     def _sdpa_wrapper(self, orig):
# #         def wrapped(query, key, value, *args, **kwargs):
# #             def _norm_shape(t):
# #                 if t.dim() == 4:  # (B,H,L,Dh) 常见
# #                     return int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
# #                 elif t.dim() == 3:  # (B,L,D) 未分头，保守按 H=1
# #                     return int(t.shape[0]), 1, int(t.shape[1]), int(t.shape[2])
# #                 else:
# #                     return 1, 1, 1, int(t.shape[-1])
# #             Bq, Hq, Lq, Dhq = _norm_shape(query)
# #             Bk, Hk, Lk, Dhk = _norm_shape(key)
# #             H = max(Hq, Hk)
# #             Dh = min(Dhq, Dhk)
# #             # QK^T + A@V（softmax忽略）
# #             self.add(Bq * H * Lq * Lk * Dh)  # QK^T
# #             self.add(Bq * H * Lq * Lk * Dh)  # A@V
# #             return orig(query, key, value, *args, **kwargs)
# #         return wrapped
# #
# #     @contextmanager
# #     def activate(self, model: nn.Module):
# #         try:
# #             # 注册线性/卷积 hook
# #             for m in model.modules():
# #                 if isinstance(m, nn.Linear):
# #                     self.handles.append(m.register_forward_hook(self._linear_hook))
# #                 elif isinstance(m, nn.Conv2d):
# #                     self.handles.append(m.register_forward_hook(self._conv2d_hook))
# #             # 替换 SDPA
# #             if hasattr(F, "scaled_dot_product_attention"):
# #                 self._orig_sdpa = F.scaled_dot_product_attention
# #                 F.scaled_dot_product_attention = self._sdpa_wrapper(self._orig_sdpa)
# #             yield
# #         finally:
# #             for h in self.handles:
# #                 h.remove()
# #             self.handles.clear()
# #             if self._orig_sdpa is not None:
# #                 F.scaled_dot_product_attention = self._orig_sdpa
# #                 self._orig_sdpa = None
# # # ============================================================
# #
# # logger = logging.getLogger(__name__)
# # max_bleus = [0.0] * 4
# #
# # tokenizer = PTBTokenizer()
# # scorers = [
# #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
# #     (Meteor(), "METEOR"),
# #     (Rouge(), "ROUGE_L"),
# #     (Cider(), "CIDEr"),
# # ]
# #
# # max_global_step = 200000000
# #
# #
# # def train(
# #     model,
# #     model_without_ddp,
# #     train_loaders,
# #     val_loaders,
# #     optimizer,
# #     epoch,
# #     global_step,
# #     device,
# #     scheduler,
# #     scaler,
# #     config,
# #     do_eval=True,
# # ):
# #     model.train()
# #     model_without_ddp.llama_model.config.use_cache = False
# #
# #     metric_logger = MetricLogger(delimiter="  ")
# #     eval_metric_logger = MetricLogger(delimiter="  ")
# #     metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
# #     loss_names = ["loss", "obj_norm", "obj_img_norm", "objid_norm", "scene_norm"]
# #     media_types = get_media_types(train_loaders)
# #
# #     for name in loss_names:
# #         metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.6f}"))
# #
# #     header = f"Train Epoch: [{epoch}]"
# #     log_freq = config.log_freq
# #
# #     if config.distributed:
# #         for d in train_loaders:
# #             d.sampler.set_epoch(epoch)
# #     train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))
# #
# #     accum_iter = config.grad_accum_steps  # 1
# #     print(f"TrainLoader Length: {len(train_loader)}")
# #     eval_freq = len(train_loader)  # 2000
# #
# #     optimizer.zero_grad()
# #     iterator = metric_logger.log_every(train_loader, log_freq, header)
# #     for i, (media_type, batch) in enumerate(iterator):
# #         for k in batch.keys():
# #             if type(batch[k]) == torch.Tensor:
# #                 batch[k] = batch[k].to(device)
# #         loss_dict = model(**batch)
# #         loss = loss_dict["loss"] / accum_iter
# #
# #         model.require_backward_grad_sync = (i + 1) % accum_iter == 0
# #
# #         scaler.scale(loss).backward()
# #
# #         if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
# #             if config.optimizer.max_grad_norm > 0:
# #                 scaler.unscale_(optimizer)
# #                 torch.nn.utils.clip_grad_norm_(
# #                     model.parameters(), config.optimizer.max_grad_norm
# #                 )
# #             scaler.step(optimizer)
# #             optimizer.zero_grad()
# #             scaler.update()
# #         scheduler.step()
# #
# #         # logging
# #         for name in loss_names:
# #             if name not in loss_dict:
# #                 continue
# #             value = loss_dict[name]
# #             value = value if isinstance(value, float) else value.item()
# #             metric_logger.update(**{f"{name}": value})
# #         metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
# #
# #         if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
# #             logs = metric_logger.get_avg_dict()
# #             log_dict_to_wandb(logs, step=global_step, prefix="train/")
# #
# #         global_step += 1
# #
# #         if do_eval and (
# #             (i + 1) % eval_freq == 0
# #             and (len(train_loader) - i >= eval_freq)
# #             or i == len(train_loader) - 1
# #         ):
# #             val_metrics = evaluate_all(
# #                 model, model_without_ddp, val_loaders, epoch, global_step, device, config
# #             )
# #             if is_main_process():
# #                 for k, v in val_metrics.items():
# #                     if k not in eval_metric_logger.meters:
# #                         eval_metric_logger.add_meter(
# #                             k, SmoothedValue(window=1, fmt="{value:.4f}")
# #                         )
# #                 eval_metric_logger.update(**val_metrics)
# #             if is_main_process() and config.wandb.enable:
# #                 logs = eval_metric_logger.get_avg_dict()
# #                 log_dict_to_wandb(logs, step=global_step, prefix="val/")
# #
# #             if is_main_process():
# #                 param_grad_dic = {
# #                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
# #                 }
# #                 state_dict = model_without_ddp.state_dict()
# #                 for k in list(state_dict.keys()):
# #                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
# #                         del state_dict[k]
# #                 save_obj = {
# #                     "model": state_dict,
# #                     "config": config,
# #                     "epoch": epoch,
# #                     "global_step": global_step,
# #                 }
# #                 if i != len(train_loader) - 1 and config.do_save and not config.debug:
# #                     torch.save(
# #                         save_obj,
# #                         join(config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"),
# #                     )
# #         if global_step > max_global_step:
# #             return global_step
# #
# #     metric_logger.synchronize_between_processes()
# #     logger.info(f"Averaged stats: {metric_logger.global_avg()}")
# #     return global_step
# #
# #
# # def evaluate_all(
# #     model, model_without_ddp, val_loaders, epoch, global_step, device, config
# # ):
# #     logger.info("Start evaluating...")
# #     model.eval()
# #     val_scores = {}
# #     for val_loader in val_loaders:
# #         new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config)
# #         val_scores = {**val_scores, **new_val_scores}
# #
# #     logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
# #     for k, v in val_scores.items():
# #         logger.info(f"{k}: {v}")
# #
# #     if is_main_process() and getattr(config, "do_save", True):
# #         param_grad_dic = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
# #         state_dict = model_without_ddp.state_dict()
# #         for k in list(state_dict.keys()):
# #             if k in param_grad_dic.keys() and not param_grad_dic[k]:
# #                 del state_dict[k]
# #         save_obj = {
# #             "model": state_dict,
# #             "config": config,
# #             "epoch": epoch,
# #             "global_step": global_step,
# #         }
# #         torch.save(save_obj, join(config.output_dir, f"ckpt_eval_{epoch:02d}_{global_step}.pth"))
# #         logger.info(
# #             f"[EVAL SAVE] Saved checkpoint to {join(config.output_dir, f'ckpt_eval_{epoch:02d}_{global_step}.pth')}")
# #
# #     model.train()
# #     return val_scores
# #
# #
# # def evaluate(model, val_loader, epoch, global_step, device, config):
# #     eval_name = val_loader.dataset.datasets[0].dataset_name
# #     logger.info(f"Evaluating {eval_name}...")
# #     if config.distributed:
# #         val_loader.sampler.set_epoch(epoch)
# #
# #     sample_freq = len(val_loader) // 5 + 1
# #     save_preds = []
# #     logger.info(f"batch-size={val_loader.batch_size} length(#batches)={len(val_loader)}")
# #
# #     # ======= [新增] 在推理循环中统计真实 MACs（≈ FLOPs） =======
# #     flops_meter = _FlopsMeter()
# #     total_macs = 0
# #     num_batches = 0
# #     # ========================================================
# #
# #     for i, batch in tqdm(enumerate(val_loader)):
# #         for k in batch.keys():
# #             if type(batch[k]) == torch.Tensor:
# #                 batch[k] = batch[k].to(device)
# #
# #         # —— 包裹一次真实前向；若你在 forward 中做了 token 裁剪，这里会反映到 FLOPs ——
# #         flops_meter.reset()
# #         with torch.no_grad():
# #             with flops_meter.activate(model):
# #                 pred = model(**batch, is_eval=True)
# #         batch_macs = flops_meter.macs
# #         total_macs += batch_macs
# #         num_batches += 1
# #
# #         # if is_main_process():
# #         #     print(f"[MACs] eval={eval_name}  batch={i}  macs={batch_macs/1e9:.3f} G  (≈ FLOPs)")
# #
# #         if "custom_prompt" in batch:
# #             batch_size = len(pred)
# #             for bi in range(batch_size):
# #                 scene_id = batch["scene_id"][bi]
# #                 obj_id = int(batch["obj_ids"][bi])
# #                 qid = batch["qid"][bi]
# #                 prompt = batch["custom_prompt"][bi]
# #                 pred_id = int(batch["pred_ids"][bi])
# #                 type_info = batch["type_infos"][bi]
# #                 tmp_pred = pred[bi]
# #                 save_preds.append(
# #                     {
# #                         "scene_id": scene_id,
# #                         "gt_id": obj_id,
# #                         "pred_id": pred_id,
# #                         "qid": qid,
# #                         "prompt": prompt,
# #                         "pred": tmp_pred,
# #                         "ref_captions": batch["ref_captions"][bi],
# #                         "type_info": type_info,
# #                     }
# #                 )
# #
# #     # 最终打印平均 MACs
# #     if is_main_process() and num_batches > 0:
# #         avg_macs = total_macs / num_batches
# #         print("=" * 64)
# #         print(f"[MACs/Average] eval={eval_name}  avg_macs={avg_macs/1e9:.3f} G  over {num_batches} batches")
# #         print("（若需严格 FLOPs≈2×MACs，可把上面的数值 ×2）")
# #         print("=" * 64)
# #
# #     dist.barrier()
# #     if len(save_preds) > 0:
# #         save_preds = sorted(
# #             save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
# #         )
# #         with open(
# #             os.path.join(
# #                 config.output_dir,
# #                 f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json",
# #             ),
# #             "w",
# #         ) as f:
# #             json.dump(save_preds, f, indent=4)
# #
# #     dist.barrier()
# #     if is_main_process():
# #         save_preds = []
# #         for rank in range(config.gpu_num):
# #             path = os.path.join(
# #                 config.output_dir,
# #                 f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json",
# #             )
# #             if os.path.exists(path):
# #                 preds = json.load(open(path, "r"))
# #                 save_preds += preds
# #                 os.remove(path)
# #         save_preds = sorted(
# #             save_preds, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
# #         )
# #         with open(
# #             os.path.join(
# #                 config.output_dir,
# #                 f"preds_epoch{epoch}_step{global_step}_{eval_name}.json",
# #             ),
# #             "w",
# #         ) as f:
# #             json.dump(save_preds, f, indent=4)
# #
# #     val_scores = {}
# #     if is_main_process() and len(save_preds) > 0:
# #         if eval_name == "scanqa":
# #             val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
# #         elif eval_name == "scanrefer":
# #             val_scores = calc_scanrefer_score(save_preds, config)
# #         elif eval_name in ["scan2cap", "scan2cap_location"]:
# #             val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
# #         elif eval_name in ["sqa3d", "sqa3d_val"]:
# #             val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config)
# #         elif eval_name == "multi3dref":
# #             val_scores = calc_multi3dref_score(save_preds, config)
# #         elif eval_name in ["nr3d", "sr3d"]:
# #             val_scores = calc_referit3d_score(save_preds, eval_name, config)
# #         elif eval_name == "scanrefer_location":
# #             val_scores = calc_scanrefer_location_score(save_preds, config)
# #         elif eval_name == "multi3dref_location":
# #             val_score = calc_multi3dref_location_score(save_preds, config)
# #         elif eval_name in ["scanrefer_test", "scan2cap_test"]:
# #             pass
# #         print(json.dumps(val_scores, indent=4))
# #     return val_scores
# #
# #
# # def setup_dataloaders(config):
# #     # train datasets, create a list of data loaders
# #     train_datasets, val_datasets = create_dataset(config)
# #
# #     if config.distributed:
# #         num_tasks = get_world_size()
# #         global_rank = get_rank()
# #         train_samplers = create_sampler(
# #             train_datasets, [True] * len(train_datasets), num_tasks, global_rank
# #         )
# #         val_samplers = create_sampler(
# #             val_datasets, [False] * len(val_datasets), num_tasks, global_rank
# #         )
# #     else:
# #         train_samplers = [None] * len(train_datasets)
# #         val_samplers = [None] * len(val_datasets)
# #
# #     train_loaders = create_loader(
# #         train_datasets,
# #         train_samplers,
# #         batch_size=[config.batch_size] * len(val_datasets),
# #         num_workers=[config.num_workers] * len(train_datasets),
# #         is_trains=[True] * len(train_datasets),
# #         collate_fns=[train_collate_fn] * len(train_datasets),
# #     )
# #     _val_collate_fn = partial(
# #         val_collate_fn, use_external_attn_maps=config.use_external_attn_maps
# #     )
# #     val_loaders = create_loader(
# #         val_datasets,
# #         val_samplers,
# #         batch_size=[config.batch_size] * len(val_datasets),
# #         num_workers=[config.num_workers] * len(val_datasets),
# #         is_trains=[False] * len(val_datasets),
# #         collate_fns=[_val_collate_fn] * len(val_datasets),
# #     )
# #
# #     return train_loaders, val_loaders
# #
# #
# # def main(config):
# #     if is_main_process() and config.wandb.enable:
# #         run = setup_wandb(config)
# #
# #     # torch.autograd.set_detect_anomaly(True)
# #     setup_seed(config.seed + get_rank())
# #     device = torch.device(config.device)
# #
# #     train_loaders, val_loaders = setup_dataloaders(config)
# #
# #     num_steps_per_epoch = sum(len(d) for d in train_loaders)
# #     config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
# #     config.scheduler.num_warmup_steps = (
# #         num_steps_per_epoch * config.scheduler.warmup_epochs
# #     )
# #     torch.backends.cudnn.benchmark = True
# #
# #     model_cls = eval(config.model.get("model_cls", "Chat3D"))
# #     (
# #         model,
# #         model_without_ddp,
# #         optimizer,
# #         scheduler,
# #         scaler,
# #         start_epoch,
# #         global_step,
# #     ) = setup_model(
# #         config,
# #         model_cls=model_cls,
# #         find_unused_parameters=True,
# #     )
# #     if is_main_process() and config.wandb.enable:
# #         wandb.watch(model)
# #
# #     save_step_interval = 1
# #     start_time = time.time()
# #     if not config.evaluate:
# #         logger.info("Start training")
# #         for epoch in range(start_epoch, config.scheduler.epochs):
# #             global_step = train(
# #                 model,
# #                 model_without_ddp,
# #                 train_loaders,
# #                 val_loaders,
# #                 optimizer,
# #                 epoch,
# #                 global_step,
# #                 device,
# #                 scheduler,
# #                 scaler,
# #                 config,
# #                 do_eval=config.do_eval,
# #             )
# #             if is_main_process():
# #                 logger.info(f"Epoch {epoch}")
# #                 param_grad_dic = {
# #                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
# #                 }
# #                 state_dict = model_without_ddp.state_dict()
# #                 for k in list(state_dict.keys()):
# #                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
# #                         del state_dict[k]
# #                 save_obj = {
# #                     "model": state_dict,
# #                     "config": config,
# #                     "epoch": epoch,
# #                     "global_step": global_step,
# #                 }
# #                 if (
# #                     (
# #                         (epoch + 1) % save_step_interval == 0
# #                         or epoch == config.scheduler.epochs - 1
# #                     )
# #                     and config.do_save
# #                     and not config.debug
# #                 ):
# #                     if config.get("save_latest", False):
# #                         torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
# #                     else:
# #                         torch.save(
# #                             save_obj,
# #                             join(
# #                                 config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"
# #                             ),
# #                         )
# #
# #             if global_step > max_global_step:
# #                 break
# #             dist.barrier()
# #
# #     if config.evaluate:
# #         evaluate_all(
# #             model,
# #             model_without_ddp,
# #             val_loaders,
# #             start_epoch - 1,
# #             global_step,
# #             device,
# #             config,
# #         )
# #
# #     total_time = time.time() - start_time
# #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
# #     logger.info(f"Training time {total_time_str}")
# #     logger.info(f"Checkpoints and Logs saved at {config.output_dir}")
# #
# #     if is_main_process() and config.wandb.enable:
# #         run.finish()
# #
# #
# # if __name__ == "__main__":
# #     cfg = setup_main()
# #     main(cfg)
# import datetime
# import logging
# import time
# from os.path import join

# import pandas as pd
# import torch
# import torch.distributed as dist
# import wandb
# from torch.utils.data import ConcatDataset
# from functools import partial

# from dataset import MetaLoader, create_dataset, create_loader, create_sampler
# from dataset.dataset_train import train_collate_fn
# from dataset.dataset_val import val_collate_fn

# from models.chat3d_fast_gt_attn import Chat3D

# from tasks.shared_utils import get_media_types, setup_model
# from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
# from utils.config_utils import setup_main
# from utils.distributed import get_rank, get_world_size, is_main_process
# from utils.logger import log_dict_to_wandb, setup_wandb
# from utils.eval import (
#     calc_scanrefer_score,
#     clean_answer,
#     calc_scan2cap_score,
#     calc_scanqa_score,
#     calc_sqa3d_score,
#     calc_multi3dref_score,
#     calc_referit3d_score,
#     calc_scanrefer_location_score,
#     calc_multi3dref_location_score,
# )

# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# import numpy as np
# from tqdm import tqdm

# import json
# import os
# import csv

# # ========== 运行时 FLOPs 计数器（MACs 口径 ≈ FLOPs） ==========
# import torch.nn as nn
# import torch.nn.functional as F
# from contextlib import contextmanager

# class _FlopsMeter:
#     """
#     推理期实时 FLOPs 计数器：
#     - 统计 nn.Linear、nn.Conv2d 的 MACs
#     - 包裹 PyTorch 2.x 的 fused scaled_dot_product_attention（SDPA）
#     - 仅在 activate() 上下文内生效，不污染训练
#     口径：MACs（乘加次数），行业常直接称为 FLOPs。若要严格 FLOPs≈2×MACs，可自行乘2。
#     """
#     def __init__(self):
#         self.macs = 0
#         self.handles = []
#         self._orig_sdpa = None

#     def reset(self):
#         self.macs = 0

#     def add(self, v: int):
#         if v is not None:
#             self.macs += int(v)

#     # 线性层：B*...*in*out
#     def _linear_hook(self, mod: nn.Linear, inp, out):
#         x = inp[0]
#         if x is None:
#             return
#         in_features = mod.in_features
#         out_features = mod.out_features
#         elems = 1
#         for d in x.shape[:-1]:
#             elems *= int(d)
#         self.add(elems * in_features * out_features)

#     # 卷积层：B * Cout * Hout * Wout * (Cin/groups * Kh * Kw)
#     def _conv2d_hook(self, mod: nn.Conv2d, inp, out):
#         x = inp[0]
#         if x is None:
#             return
#         B = int(x.shape[0])
#         Cout = int(out.shape[1])
#         Hout = int(out.shape[2])
#         Wout = int(out.shape[3])
#         Cin = int(mod.in_channels)
#         if isinstance(mod.kernel_size, tuple):
#             Kh, Kw = mod.kernel_size
#         else:
#             Kh = Kw = int(mod.kernel_size)
#         groups = int(mod.groups)
#         macs_per_out = (Cin // groups) * Kh * Kw
#         self.add(B * Cout * Hout * Wout * macs_per_out)

#     # 包裹 SDPA，依据 q/k/v 的真实形状统计注意力核心 FLOPs
#     def _sdpa_wrapper(self, orig):
#         def wrapped(query, key, value, *args, **kwargs):
#             def _norm_shape(t):
#                 if t.dim() == 4:  # (B,H,L,Dh)
#                     return int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
#                 elif t.dim() == 3:  # (B,L,D)
#                     return int(t.shape[0]), 1, int(t.shape[1]), int(t.shape[2])
#                 else:
#                     return 1, 1, 1, int(t.shape[-1])
#             Bq, Hq, Lq, Dhq = _norm_shape(query)
#             Bk, Hk, Lk, Dhk = _norm_shape(key)
#             H = max(Hq, Hk)
#             Dh = min(Dhq, Dhk)
#             # QK^T + A@V（softmax忽略）
#             self.add(Bq * H * Lq * Lk * Dh)  # QK^T
#             self.add(Bq * H * Lq * Lk * Dh)  # A@V
#             return orig(query, key, value, *args, **kwargs)
#         return wrapped

#     @contextmanager
#     def activate(self, model: nn.Module):
#         try:
#             # 注册线性/卷积 hook
#             for m in model.modules():
#                 if isinstance(m, nn.Linear):
#                     self.handles.append(m.register_forward_hook(self._linear_hook))
#                 elif isinstance(m, nn.Conv2d):
#                     self.handles.append(m.register_forward_hook(self._conv2d_hook))
#             # 替换 SDPA
#             if hasattr(F, "scaled_dot_product_attention"):
#                 self._orig_sdpa = F.scaled_dot_product_attention
#                 F.scaled_dot_product_attention = self._sdpa_wrapper(self._orig_sdpa)
#             yield
#         finally:
#             for h in self.handles:
#                 h.remove()
#             self.handles.clear()
#             if self._orig_sdpa is not None:
#                 F.scaled_dot_product_attention = self._orig_sdpa
#                 self._orig_sdpa = None
# # ============================================================

# logger = logging.getLogger(__name__)
# max_bleus = [0.0] * 4

# tokenizer = PTBTokenizer()
# scorers = [
#     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#     (Meteor(), "METEOR"),
#     (Rouge(), "ROUGE_L"),
#     (Cider(), "CIDEr"),
# ]

# max_global_step = 200000000


# def train(
#     model,
#     model_without_ddp,
#     train_loaders,
#     val_loaders,
#     optimizer,
#     epoch,
#     global_step,
#     device,
#     scheduler,
#     scaler,
#     config,
#     do_eval=True,
# ):
#     model.train()
#     model_without_ddp.llama_model.config.use_cache = False

#     metric_logger = MetricLogger(delimiter="  ")
#     eval_metric_logger = MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
#     loss_names = ["loss", "obj_norm", "obj_img_norm", "objid_norm", "scene_norm"]
#     media_types = get_media_types(train_loaders)

#     for name in loss_names:
#         metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.6f}"))

#     header = f"Train Epoch: [{epoch}]"
#     log_freq = config.log_freq

#     if config.distributed:
#         for d in train_loaders:
#             d.sampler.set_epoch(epoch)
#     train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

#     accum_iter = config.grad_accum_steps  # 1
#     print(f"TrainLoader Length: {len(train_loader)}")
#     eval_freq = len(train_loader)  # 2000

#     optimizer.zero_grad()
#     iterator = metric_logger.log_every(train_loader, log_freq, header)
#     for i, (media_type, batch) in enumerate(iterator):
#         for k in batch.keys():
#             if isinstance(batch[k], torch.Tensor):
#                 batch[k] = batch[k].to(device)
#         loss_dict = model(**batch)
#         loss = loss_dict["loss"] / accum_iter

#         model.require_backward_grad_sync = (i + 1) % accum_iter == 0

#         scaler.scale(loss).backward()

#         if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
#             if config.optimizer.max_grad_norm > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), config.optimizer.max_grad_norm
#                 )
#             scaler.step(optimizer)
#             optimizer.zero_grad()
#             scaler.update()
#         scheduler.step()

#         # logging
#         for name in loss_names:
#             if name not in loss_dict:
#                 continue
#             value = loss_dict[name]
#             value = value if isinstance(value, float) else value.item()
#             metric_logger.update(**{f"{name}": value})
#         metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

#         if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
#             logs = metric_logger.get_avg_dict()
#             log_dict_to_wandb(logs, step=global_step, prefix="train/")

#         global_step += 1

#         if do_eval and (
#             (i + 1) % eval_freq == 0
#             and (len(train_loader) - i >= eval_freq)
#             or i == len(train_loader) - 1
#         ):
#             val_metrics = evaluate_all(
#                 model, model_without_ddp, val_loaders, epoch, global_step, device, config
#             )
#             if is_main_process():
#                 for k, v in val_metrics.items():
#                     if k not in eval_metric_logger.meters:
#                         eval_metric_logger.add_meter(
#                             k, SmoothedValue(window=1, fmt="{value:.4f}")
#                         )
#                 eval_metric_logger.update(**val_metrics)
#             if is_main_process() and config.wandb.enable:
#                 logs = eval_metric_logger.get_avg_dict()
#                 log_dict_to_wandb(logs, step=global_step, prefix="val/")

#             if is_main_process():
#                 param_grad_dic = {
#                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
#                 }
#                 state_dict = model_without_ddp.state_dict()
#                 for k in list(state_dict.keys()):
#                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
#                         del state_dict[k]
#                 save_obj = {
#                     "model": state_dict,
#                     "config": config,
#                     "epoch": epoch,
#                     "global_step": global_step,
#                 }
#                 if i != len(train_loader) - 1 and config.do_save and not config.debug:
#                     torch.save(
#                         save_obj,
#                         join(config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"),
#                     )
#         if global_step > max_global_step:
#             return global_step

#     metric_logger.synchronize_between_processes()
#     logger.info(f"Averaged stats: {metric_logger.global_avg()}")
#     return global_step


# def evaluate_all(
#     model, model_without_ddp, val_loaders, epoch, global_step, device, config
# ):
#     """
#     针对多个验证集分别评测，每个数据集单独保存预测与分析结果
#     """
#     logger.info("Start evaluating...")
#     model.eval()
#     val_scores = {}

#     for val_loader in val_loaders:
#         eval_name = val_loader.dataset.datasets[0].dataset_name
#         logger.info(f"--- Start evaluating dataset [{eval_name}] ---")
#         new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config, eval_name)
#         val_scores = {**val_scores, **new_val_scores}
#         logger.info(f"--- Finished evaluating [{eval_name}] ---\n")

#     logger.info(f"[epoch={epoch}, global steps={global_step}] All Val Results:")
#     for k, v in val_scores.items():
#         logger.info(f"{k}: {v}")

#     if is_main_process() and getattr(config, "do_save", True):
#         param_grad_dic = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
#         state_dict = model_without_ddp.state_dict()
#         for k in list(state_dict.keys()):
#             if k in param_grad_dic.keys() and not param_grad_dic[k]:
#                 del state_dict[k]
#         save_obj = {
#             "model": state_dict,
#             "config": config,
#             "epoch": epoch,
#             "global_step": global_step,
#         }
#         ckpt_path = join(config.output_dir, f"ckpt_eval_{epoch:02d}_{global_step}.pth")
#         torch.save(save_obj, ckpt_path)
#         logger.info(f"[EVAL SAVE] Saved checkpoint to {ckpt_path}")

#     model.train()
#     return val_scores


# # =================== 样本级对错拆分工具函数 =================== #
# def _split_correct_wrong(eval_name: str, samples: list):
#     """
#     给每条样本打上 is_correct，并按数据集类型拆分为 correct / wrong 两份列表。
#     - grounding/指代: pred_id == gt_id
#     - QA: exact match (clean_answer 之后，匹配任一 GT)
#     - caption: 无二元对错 -> is_correct 置为 None，不纳入 correct/wrong
#     """
#     correct, wrong = [], []
#     for s in samples:
#         ok = None

#         # —— Grounding / 指代类 —— #
#         if eval_name in ["scanrefer", "scanrefer_location",
#                          "multi3dref", "multi3dref_location",
#                          "nr3d", "sr3d"]:
#             try:
#                 ok = int(s.get("pred_id", -1)) == int(s.get("gt_id", -2))
#             except Exception:
#                 ok = False

#         # —— QA 类（严格 EM）—— #
#         elif eval_name in ["scanqa", "sqa3d", "sqa3d_val"]:
#             pred = clean_answer(s.get("pred", ""))
#             gt_list = s.get("gt_answers", s.get("answers", s.get("answer", [])))
#             if isinstance(gt_list, str):
#                 gt_list = [gt_list]
#             if isinstance(gt_list, list):
#                 gt_clean = [clean_answer(x) for x in gt_list]
#                 ok = (pred in set(gt_clean))
#             else:
#                 ok = False

#         # —— Caption 类：不给二元标签 —— #
#         elif eval_name in ["scan2cap", "scan2cap_location",
#                            "scanrefer_test", "scan2cap_test"]:
#             ok = None

#         s_with_flag = {**s, "is_correct": ok}

#         if ok is True:
#             correct.append(s_with_flag)
#         elif ok is False:
#             wrong.append(s_with_flag)
#         # ok is None -> 跳过二元划分（比如 caption）

#     return correct, wrong
# # ================================================================ #


# def evaluate(model, val_loader, epoch, global_step, device, config, eval_name=None):
#     """
#     针对单个数据集评测，保存预测文件、对错拆分、CSV（按数据集名分别保存）
#     """
#     if eval_name is None:
#         eval_name = val_loader.dataset.datasets[0].dataset_name
#     logger.info(f"Evaluating {eval_name}...")

#     if config.distributed:
#         val_loader.sampler.set_epoch(epoch)

#     save_preds_rank = []
#     flops_meter = _FlopsMeter()
#     total_macs, num_batches = 0, 0

#     for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
#         for k in batch.keys():
#             if isinstance(batch[k], torch.Tensor):
#                 batch[k] = batch[k].to(device)

#         flops_meter.reset()
#         with torch.no_grad():
#             with flops_meter.activate(model):
#                 pred = model(**batch, is_eval=True)
#         total_macs += flops_meter.macs
#         num_batches += 1

#         if "custom_prompt" in batch:
#             for bi in range(len(pred)):
#                 rec = {
#                     "scene_id": batch["scene_id"][bi],
#                     "gt_id": int(batch["obj_ids"][bi]),
#                     "pred_id": int(batch["pred_ids"][bi]),
#                     "qid": batch["qid"][bi],
#                     "prompt": batch["custom_prompt"][bi],
#                     "pred": pred[bi],
#                     "ref_captions": batch["ref_captions"][bi],
#                     "type_info": batch["type_infos"][bi],
#                 }
#                 # 可选：GT答案
#                 if "answers" in batch and len(batch["answers"]) > bi:
#                     rec["gt_answers"] = batch["answers"][bi]
#                 elif "answer" in batch and len(batch["answer"]) > bi:
#                     rec["gt_answers"] = batch["answer"][bi]
#                 save_preds_rank.append(rec)

#     # ===== 打印 FLOPs 统计 =====
#     if is_main_process() and num_batches > 0:
#         avg_macs = total_macs / num_batches
#         print("=" * 64)
#         print(f"[MACs/Average] {eval_name}: {avg_macs/1e9:.3f} G  over {num_batches} batches")
#         print("（若需严格 FLOPs≈2×MACs，可把上面的数值 ×2）")
#         print("=" * 64)

#     # ===== 各 rank 各自写临时文件 =====
#     dist.barrier()
#     if len(save_preds_rank) > 0:
#         save_preds_rank = sorted(
#             save_preds_rank, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
#         )
#         tmp_rank_json = os.path.join(
#             config.output_dir,
#             f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json",
#         )
#         with open(tmp_rank_json, "w") as f:
#             json.dump(save_preds_rank, f, indent=4)

#     # ===== 主进程合并为单个数据集文件，并导出对/错 & CSV =====
#     dist.barrier()
#     if is_main_process():
#         merged = []
#         for rank in range(config.gpu_num):
#             path = os.path.join(
#                 config.output_dir,
#                 f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json",
#             )
#             if os.path.exists(path):
#                 preds = json.load(open(path, "r"))
#                 merged += preds
#                 os.remove(path)
#         merged = sorted(
#             merged, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
#         )

#         # 1) 写全量预测
#         merged_json_path = os.path.join(
#             config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"
#         )
#         with open(merged_json_path, "w") as f:
#             json.dump(merged, f, indent=4)

#         # 2) 拆分对/错
#         if len(merged) > 0:
#             correct, wrong = _split_correct_wrong(eval_name, merged)

#             with open(os.path.join(config.output_dir, f"correct_{eval_name}.json"), "w") as f:
#                 json.dump(correct, f, indent=4)
#             with open(os.path.join(config.output_dir, f"wrong_{eval_name}.json"), "w") as f:
#                 json.dump(wrong, f, indent=4)

#             # 3) 导出 CSV（scene_id/qid/gt_id/pred_id/pred/is_correct）
#             csv_path = os.path.join(config.output_dir, f"summary_{eval_name}.csv")
#             keys = ["scene_id", "qid", "gt_id", "pred_id", "pred", "is_correct"]
#             with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
#                 writer = csv.writer(fcsv)
#                 writer.writerow(keys)
#                 for s in merged:
#                     writer.writerow([
#                         s.get("scene_id", ""),
#                         s.get("qid", ""),
#                         s.get("gt_id", ""),
#                         s.get("pred_id", ""),
#                         s.get("pred", ""),
#                         s.get("is_correct", "")
#                     ])

#             # 控制台摘要
#             num_c = sum(1 for s in merged if s.get("is_correct") is True)
#             num_w = sum(1 for s in merged if s.get("is_correct") is False)
#             num_n = len(merged) - num_c - num_w  # caption 等无二元定义
#             print(f"[{eval_name}] correct={num_c}  wrong={num_w}  undecidable={num_n}")

#     # ===== 计算并返回该数据集的指标 =====
#     val_scores = {}
#     # 注意：下面的 save list 应该是“主进程合并后的 merged”，但为了结构一致，
#     # 在分布式下只在主进程计算指标并广播结果更稳妥。这里保持与原始风格一致：
#     if is_main_process():
#         preds_path = os.path.join(
#             config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"
#         )
#         if os.path.exists(preds_path):
#             save_preds = json.load(open(preds_path, "r"))
#             if eval_name == "scanqa":
#                 val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
#             elif eval_name == "scanrefer":
#                 val_scores = calc_scanrefer_score(save_preds, config)
#             elif eval_name in ["scan2cap", "scan2cap_location"]:
#                 val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
#             elif eval_name in ["sqa3d", "sqa3d_val"]:
#                 val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config)
#             elif eval_name == "multi3dref":
#                 val_scores = calc_multi3dref_score(save_preds, config)
#             elif eval_name in ["nr3d", "sr3d"]:
#                 val_scores = calc_referit3d_score(save_preds, eval_name, config)
#             elif eval_name == "scanrefer_location":
#                 val_scores = calc_scanrefer_location_score(save_preds, config)
#             elif eval_name == "multi3dref_location":
#                 val_scores = calc_multi3dref_location_score(save_preds, config)
#             elif eval_name in ["scanrefer_test", "scan2cap_test"]:
#                 pass
#             print(json.dumps(val_scores, indent=4))

#     # 可以在此处用 dist 广播 val_scores（若需要其它 rank 也拿到数值）
#     return {f"{eval_name}/{k}": v for k, v in val_scores.items()}


# def setup_dataloaders(config):
#     # train & val datasets
#     train_datasets, val_datasets = create_dataset(config)

#     if config.distributed:
#         num_tasks = get_world_size()
#         global_rank = get_rank()
#         train_samplers = create_sampler(
#             train_datasets, [True] * len(train_datasets), num_tasks, global_rank
#         )
#         val_samplers = create_sampler(
#             val_datasets, [False] * len(val_datasets), num_tasks, global_rank
#         )
#     else:
#         train_samplers = [None] * len(train_datasets)
#         val_samplers = [None] * len(val_datasets)

#     # ✅ 修正长度：按 train_datasets 的数量
#     train_loaders = create_loader(
#         train_datasets,
#         train_samplers,
#         batch_size=[config.batch_size] * len(train_datasets),
#         num_workers=[config.num_workers] * len(train_datasets),
#         is_trains=[True] * len(train_datasets),
#         collate_fns=[train_collate_fn] * len(train_datasets),
#     )

#     _val_collate_fn = partial(
#         val_collate_fn, use_external_attn_maps=config.use_external_attn_maps
#     )
#     val_loaders = create_loader(
#         val_datasets,
#         val_samplers,
#         batch_size=[config.batch_size] * len(val_datasets),
#         num_workers=[config.num_workers] * len(val_datasets),
#         is_trains=[False] * len(val_datasets),
#         collate_fns=[_val_collate_fn] * len(val_datasets),  # ✅ 修正写法
#     )

#     return train_loaders, val_loaders


# def main(config):
#     if is_main_process() and config.wandb.enable:
#         run = setup_wandb(config)

#     setup_seed(config.seed + get_rank())
#     device = torch.device(config.device)

#     train_loaders, val_loaders = setup_dataloaders(config)

#     num_steps_per_epoch = sum(len(d) for d in train_loaders)
#     config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
#     config.scheduler.num_warmup_steps = (
#         num_steps_per_epoch * config.scheduler.warmup_epochs
#     )
#     torch.backends.cudnn.benchmark = True

#     model_cls = eval(config.model.get("model_cls", "Chat3D"))
#     (
#         model,
#         model_without_ddp,
#         optimizer,
#         scheduler,
#         scaler,
#         start_epoch,
#         global_step,
#     ) = setup_model(
#         config,
#         model_cls=model_cls,
#         find_unused_parameters=True,
#     )
#     if is_main_process() and config.wandb.enable:
#         wandb.watch(model)

#     save_step_interval = 1
#     start_time = time.time()
#     if not config.evaluate:
#         logger.info("Start training")
#         for epoch in range(start_epoch, config.scheduler.epochs):
#             global_step = train(
#                 model,
#                 model_without_ddp,
#                 train_loaders,
#                 val_loaders,
#                 optimizer,
#                 epoch,
#                 global_step,
#                 device,
#                 scheduler,
#                 scaler,
#                 config,
#                 do_eval=config.do_eval,
#             )
#             if is_main_process():
#                 logger.info(f"Epoch {epoch}")
#                 param_grad_dic = {
#                     k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
#                 }
#                 state_dict = model_without_ddp.state_dict()
#                 for k in list(state_dict.keys()):
#                     if k in param_grad_dic.keys() and not param_grad_dic[k]:
#                         del state_dict[k]
#                 save_obj = {
#                     "model": state_dict,
#                     "config": config,
#                     "epoch": epoch,
#                     "global_step": global_step,
#                 }
#                 if (
#                     (
#                         (epoch + 1) % save_step_interval == 0
#                         or epoch == config.scheduler.epochs - 1
#                     )
#                     and config.do_save
#                     and not config.debug
#                 ):
#                     if config.get("save_latest", False):
#                         torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
#                     else:
#                         torch.save(
#                             save_obj,
#                             join(
#                                 config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"
#                             ),
#                         )

#             if global_step > max_global_step:
#                 break
#             dist.barrier()

#     if config.evaluate:
#         evaluate_all(
#             model,
#             model_without_ddp,
#             val_loaders,
#             start_epoch - 1,
#             global_step,
#             device,
#             config,
#         )

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     logger.info(f"Training time {total_time_str}")
#     logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

#     if is_main_process() and config.wandb.enable:
#         run.finish()


# if __name__ == "__main__":
#     cfg = setup_main()
#     main(cfg)
import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
from functools import partial

from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.dataset_train import train_collate_fn
from dataset.dataset_val import val_collate_fn

from models.chat3d_fast_gt_attn import Chat3D

from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.eval import (
    calc_scanrefer_score,
    clean_answer,
    calc_scan2cap_score,
    calc_scanqa_score,
    calc_sqa3d_score,
    calc_multi3dref_score,
    calc_referit3d_score,
    calc_scanrefer_location_score,
    calc_multi3dref_location_score,
)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import numpy as np
from tqdm import tqdm

import json
import os
import csv

# ========== 运行时 FLOPs 计数器（MACs 口径 ≈ FLOPs） ==========
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

class _FlopsMeter:
    """
    推理期实时 FLOPs 计数器：
    - 统计 nn.Linear、nn.Conv2d 的 MACs
    - 包裹 PyTorch 2.x 的 fused scaled_dot_product_attention（SDPA）
    - 仅在 activate() 上下文内生效，不污染训练
    口径：MACs（乘加次数），行业常直接称为 FLOPs。若要严格 FLOPs≈2×MACs，可自行乘2。
    """
    def __init__(self):
        self.macs = 0
        self.handles = []
        self._orig_sdpa = None

    def reset(self):
        self.macs = 0

    def add(self, v: int):
        if v is not None:
            self.macs += int(v)

    # 线性层：B*...*in*out
    def _linear_hook(self, mod: nn.Linear, inp, out):
        x = inp[0]
        if x is None:
            return
        in_features = mod.in_features
        out_features = mod.out_features
        elems = 1
        for d in x.shape[:-1]:
            elems *= int(d)
        self.add(elems * in_features * out_features)

    # 卷积层：B * Cout * Hout * Wout * (Cin/groups * Kh * Kw)
    def _conv2d_hook(self, mod: nn.Conv2d, inp, out):
        x = inp[0]
        if x is None:
            return
        B = int(x.shape[0])
        Cout = int(out.shape[1])
        Hout = int(out.shape[2])
        Wout = int(out.shape[3])
        Cin = int(mod.in_channels)
        if isinstance(mod.kernel_size, tuple):
            Kh, Kw = mod.kernel_size
        else:
            Kh = Kw = int(mod.kernel_size)
        groups = int(mod.groups)
        macs_per_out = (Cin // groups) * Kh * Kw
        self.add(B * Cout * Hout * Wout * macs_per_out)

    # 包裹 SDPA，依据 q/k/v 的真实形状统计注意力核心 FLOPs
    def _sdpa_wrapper(self, orig):
        def wrapped(query, key, value, *args, **kwargs):
            def _norm_shape(t):
                if t.dim() == 4:  # (B,H,L,Dh)
                    return int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
                elif t.dim() == 3:  # (B,L,D)
                    return int(t.shape[0]), 1, int(t.shape[1]), int(t.shape[2])
                else:
                    return 1, 1, 1, int(t.shape[-1])
            Bq, Hq, Lq, Dhq = _norm_shape(query)
            Bk, Hk, Lk, Dhk = _norm_shape(key)
            H = max(Hq, Hk)
            Dh = min(Dhq, Dhk)
            # QK^T + A@V（softmax忽略）
            self.add(Bq * H * Lq * Lk * Dh)  # QK^T
            self.add(Bq * H * Lq * Lk * Dh)  # A@V
            return orig(query, key, value, *args, **kwargs)
        return wrapped

    @contextmanager
    def activate(self, model: nn.Module):
        try:
            # 注册线性/卷积 hook
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    self.handles.append(m.register_forward_hook(self._linear_hook))
                elif isinstance(m, nn.Conv2d):
                    self.handles.append(m.register_forward_hook(self._conv2d_hook))
            # 替换 SDPA
            if hasattr(F, "scaled_dot_product_attention"):
                self._orig_sdpa = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = self._sdpa_wrapper(self._orig_sdpa)
            yield
        finally:
            for h in self.handles:
                h.remove()
            self.handles.clear()
            if self._orig_sdpa is not None:
                F.scaled_dot_product_attention = self._orig_sdpa
                self._orig_sdpa = None
# ============================================================

logger = logging.getLogger(__name__)
max_bleus = [0.0] * 4

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
]

max_global_step = 200000000


def train(
    model,
    model_without_ddp,
    train_loaders,
    val_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    do_eval=True,
):
    model.train()
    model_without_ddp.llama_model.config.use_cache = False

    metric_logger = MetricLogger(delimiter="  ")
    eval_metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss", "obj_norm", "obj_img_norm", "objid_norm", "scene_norm"]
    media_types = get_media_types(train_loaders)

    for name in loss_names:
        metric_logger.add_meter(f"{name}", SmoothedValue(window=1, fmt="{value:.6f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    accum_iter = config.grad_accum_steps  # 1
    print(f"TrainLoader Length: {len(train_loader)}")
    eval_freq = len(train_loader)  # 2000

    optimizer.zero_grad()
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, batch) in enumerate(iterator):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        loss_dict = model(**batch)
        loss = loss_dict["loss"] / accum_iter

        model.require_backward_grad_sync = (i + 1) % accum_iter == 0

        scaler.scale(loss).backward()

        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.max_grad_norm
                )
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            if name not in loss_dict:
                continue
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{name}": value})
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if do_eval and (
            (i + 1) % eval_freq == 0
            and (len(train_loader) - i >= eval_freq)
            or i == len(train_loader) - 1
        ):
            val_metrics = evaluate_all(
                model, model_without_ddp, val_loaders, epoch, global_step, device, config
            )
            if is_main_process():
                for k, v in val_metrics.items():
                    if k not in eval_metric_logger.meters:
                        eval_metric_logger.add_meter(
                            k, SmoothedValue(window=1, fmt="{value:.4f}")
                        )
                eval_metric_logger.update(**val_metrics)
            if is_main_process() and config.wandb.enable:
                logs = eval_metric_logger.get_avg_dict()
                log_dict_to_wandb(logs, step=global_step, prefix="val/")

            if is_main_process():
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                state_dict = model_without_ddp.state_dict()
                for k in list(state_dict.keys()):
                    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if i != len(train_loader) - 1 and config.do_save and not config.debug:
                    torch.save(
                        save_obj,
                        join(config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"),
                    )
        if global_step > max_global_step:
            return global_step

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def evaluate_all(
    model, model_without_ddp, val_loaders, epoch, global_step, device, config
):
    """
    针对多个验证集分别评测，每个数据集单独保存预测与分析结果
    """
    logger.info("Start evaluating...")
    model.eval()
    val_scores = {}

    for val_loader in val_loaders:
        eval_name = val_loader.dataset.datasets[0].dataset_name
        logger.info(f"--- Start evaluating dataset [{eval_name}] ---")
        new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config, eval_name)
        val_scores = {**val_scores, **new_val_scores}
        logger.info(f"--- Finished evaluating [{eval_name}] ---\n")

    logger.info(f"[epoch={epoch}, global steps={global_step}] All Val Results:")
    for k, v in val_scores.items():
        logger.info(f"{k}: {v}")

    if is_main_process() and getattr(config, "do_save", True):
        param_grad_dic = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
        state_dict = model_without_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }
        ckpt_path = join(config.output_dir, f"ckpt_eval_{epoch:02d}_{global_step}.pth")
        torch.save(save_obj, ckpt_path)
        logger.info(f"[EVAL SAVE] Saved checkpoint to {ckpt_path}")

    model.train()
    return val_scores


# =================== 样本级对错拆分工具函数 =================== #
def _split_correct_wrong(eval_name: str, samples: list):
    """
    给每条样本打上 is_correct，并按数据集类型拆分为 correct / wrong 两份列表。
    - grounding/指代: pred_id == gt_id
    - QA: exact match (clean_answer 之后，匹配任一 GT)
    - caption: 无二元对错 -> is_correct 置为 None，不纳入 correct/wrong
    """
    correct, wrong = [], []
    for s in samples:
        ok = None

        # —— Grounding / 指代类 —— #
        if eval_name in ["scanrefer", "scanrefer_location",
                         "multi3dref", "multi3dref_location",
                         "nr3d", "sr3d"]:
            try:
                ok = int(s.get("pred_id", -1)) == int(s.get("gt_id", -2))
            except Exception:
                ok = False

        # —— QA 类（严格 EM）—— #
        elif eval_name in ["scanqa", "sqa3d", "sqa3d_val"]:
            pred = clean_answer(s.get("pred", ""))
            gt_list = s.get("gt_answers", s.get("answers", s.get("answer", [])))
            if isinstance(gt_list, str):
                gt_list = [gt_list]
            if isinstance(gt_list, list):
                gt_clean = [clean_answer(x) for x in gt_list]
                ok = (pred in set(gt_clean))
            else:
                ok = False

        # —— Caption 类：不给二元标签 —— #
        elif eval_name in ["scan2cap", "scan2cap_location",
                           "scanrefer_test", "scan2cap_test"]:
            ok = None

        s_with_flag = {**s, "is_correct": ok}

        if ok is True:
            correct.append(s_with_flag)
        elif ok is False:
            wrong.append(s_with_flag)
        # ok is None -> 跳过二元划分（比如 caption）

    return correct, wrong
# ================================================================ #


def evaluate(model, val_loader, epoch, global_step, device, config, eval_name=None):
    """
    针对单个数据集评测，保存预测文件、对错拆分、CSV（按数据集名分别保存）
    同时：
      1）打印模型总参数量（Total Params）与可训练参数量（Trainable Params）
      2）用前 50 个样本估计平均 MACs（≈ FLOPs per sample）
    """
    if eval_name is None:
        eval_name = val_loader.dataset.datasets[0].dataset_name
    logger.info(f"Evaluating {eval_name}...")

    if config.distributed:
        val_loader.sampler.set_epoch(epoch)

    # ===== 只在主进程打印参数量 =====
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("=" * 64)
        print(f"[{eval_name}] Total Params:     {total_params/1e6:.3f} M")
        print(f"[{eval_name}] Trainable Params: {trainable_params/1e6:.3f} M")
        print("=" * 64)

    save_preds_rank = []
    flops_meter = _FlopsMeter()

    # 只用前 50 个样本估 FLOPs
    max_flop_samples = 50
    counted_samples = 0
    sum_macs_for_avg = 0.0
    timing_cfg = getattr(config, "timing", None)
    timing_enable = bool(getattr(timing_cfg, "enable", False)) if timing_cfg else False
    timing_warmup = int(getattr(timing_cfg, "warmup_iters", 0)) if timing_cfg else 0
    timing_total_ms = 0.0
    timing_total_samples = 0

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        flops_meter.reset()
        with torch.no_grad():
            with flops_meter.activate(model):
                if timing_enable:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    else:
                        start_time = time.perf_counter()
                pred = model(**batch, is_eval=True)
                if timing_enable and device.type == "cuda":
                    end_event.record()

        try:
            bs = len(pred)
        except TypeError:
            bs = len(batch["scene_id"])

        if timing_enable:
            if device.type == "cuda":
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if i >= timing_warmup and bs > 0:
                timing_total_ms += elapsed_ms
                timing_total_samples += bs

        # ===== FLOPs 统计：只累计前 50 个样本 =====
        if counted_samples < max_flop_samples:
            if bs > 0 and flops_meter.macs > 0:
                per_sample_macs = flops_meter.macs / float(bs)
                remain = max_flop_samples - counted_samples
                use_n = min(bs, remain)
                sum_macs_for_avg += per_sample_macs * use_n
                counted_samples += use_n

        if "custom_prompt" in batch:
            for bi in range(len(pred)):
                rec = {
                    "scene_id": batch["scene_id"][bi],
                    "gt_id": int(batch["obj_ids"][bi]),
                    "pred_id": int(batch["pred_ids"][bi]),
                    "qid": batch["qid"][bi],
                    "prompt": batch["custom_prompt"][bi],
                    "pred": pred[bi],
                    "ref_captions": batch["ref_captions"][bi],
                    "type_info": batch["type_infos"][bi],
                }
                # 可选：GT答案
                if "answers" in batch and len(batch["answers"]) > bi:
                    rec["gt_answers"] = batch["answers"][bi]
                elif "answer" in batch and len(batch["answer"]) > bi:
                    rec["gt_answers"] = batch["answer"][bi]
                save_preds_rank.append(rec)

    # ===== 打印 FLOPs 统计（只用前 50 个样本） =====
    if is_main_process() and counted_samples > 0:
        avg_macs = sum_macs_for_avg / float(counted_samples)
        print("=" * 64)
        print(f"[{eval_name}] Approx MACs per sample "
              f"(first {counted_samples} samples): {avg_macs/1e9:.3f} G")
        print("（若需严格 FLOPs≈2×MACs，可把上面的数值 ×2）")
        print("=" * 64)

    if timing_enable:
        total_ms = torch.tensor(timing_total_ms, device=device, dtype=torch.float64)
        total_samples = torch.tensor(timing_total_samples, device=device, dtype=torch.float64)
        if config.distributed:
            dist.all_reduce(total_ms, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        if is_main_process() and total_samples.item() > 0:
            avg_ms = total_ms.item() / total_samples.item()
            print("=" * 64)
            print(
                f"[{eval_name}] Avg inference time per sample: {avg_ms:.3f} ms "
                f"(warmup={timing_warmup})"
            )
            print("=" * 64)

    # ===== 各 rank 各自写临时文件 =====
    dist.barrier()
    if len(save_preds_rank) > 0:
        save_preds_rank = sorted(
            save_preds_rank, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
        )
        tmp_rank_json = os.path.join(
            config.output_dir,
            f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json",
        )
        with open(tmp_rank_json, "w") as f:
            json.dump(save_preds_rank, f, indent=4)

    # ===== 主进程合并为单个数据集文件，并导出对/错 & CSV =====
    dist.barrier()
    if is_main_process():
        merged = []
        for rank in range(config.gpu_num):
            path = os.path.join(
                config.output_dir,
                f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json",
            )
            if os.path.exists(path):
                preds = json.load(open(path, "r"))
                merged += preds
                os.remove(path)
        merged = sorted(
            merged, key=lambda x: f"{x['scene_id']}_{x['gt_id']:03}_{x['qid']}"
        )

        # 1) 写全量预测
        merged_json_path = os.path.join(
            config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"
        )
        with open(merged_json_path, "w") as f:
            json.dump(merged, f, indent=4)

        # 2) 拆分对/错
        if len(merged) > 0:
            correct, wrong = _split_correct_wrong(eval_name, merged)

            with open(os.path.join(config.output_dir, f"correct_{eval_name}.json"), "w") as f:
                json.dump(correct, f, indent=4)
            with open(os.path.join(config.output_dir, f"wrong_{eval_name}.json"), "w") as f:
                json.dump(wrong, f, indent=4)

            # 3) 导出 CSV（scene_id/qid/gt_id/pred_id/pred/is_correct）
            csv_path = os.path.join(config.output_dir, f"summary_{eval_name}.csv")
            keys = ["scene_id", "qid", "gt_id", "pred_id", "pred", "is_correct"]
            with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(keys)
                for s in merged:
                    writer.writerow([
                        s.get("scene_id", ""),
                        s.get("qid", ""),
                        s.get("gt_id", ""),
                        s.get("pred_id", ""),
                        s.get("pred", ""),
                        s.get("is_correct", "")
                    ])

            # 控制台摘要
            num_c = sum(1 for s in merged if s.get("is_correct") is True)
            num_w = sum(1 for s in merged if s.get("is_correct") is False)
            num_n = len(merged) - num_c - num_w  # caption 等无二元定义
            print(f"[{eval_name}] correct={num_c}  wrong={num_w}  undecidable={num_n}")

    # ===== 计算并返回该数据集的指标 =====
    val_scores = {}
    if is_main_process():
        preds_path = os.path.join(
            config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"
        )
        if os.path.exists(preds_path):
            save_preds = json.load(open(preds_path, "r"))
            if eval_name == "scanqa":
                val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
            elif eval_name == "scanrefer":
                val_scores = calc_scanrefer_score(save_preds, config)
            elif eval_name in ["scan2cap", "scan2cap_location"]:
                val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
            elif eval_name in ["sqa3d", "sqa3d_val"]:
                val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config)
            elif eval_name == "multi3dref":
                val_scores = calc_multi3dref_score(save_preds, config)
            elif eval_name in ["nr3d", "sr3d"]:
                val_scores = calc_referit3d_score(save_preds, eval_name, config)
            elif eval_name == "scanrefer_location":
                val_scores = calc_scanrefer_location_score(save_preds, config)
            elif eval_name == "multi3dref_location":
                val_scores = calc_multi3dref_location_score(save_preds, config)
            elif eval_name in ["scanrefer_test", "scan2cap_test"]:
                pass
            print(json.dumps(val_scores, indent=4))

    return {f"{eval_name}/{k}": v for k, v in val_scores.items()}


def setup_dataloaders(config):
    # train & val datasets
    train_datasets, val_datasets = create_dataset(config)

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [True] * len(train_datasets), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [False] * len(val_datasets), num_tasks, global_rank
        )
    else:
        train_samplers = [None] * len(train_datasets)
        val_samplers = [None] * len(val_datasets)

    # ✅ 修正长度：按 train_datasets 的数量
    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[config.batch_size] * len(train_datasets),
        num_workers=[config.num_workers] * len(train_datasets),
        is_trains=[True] * len(train_datasets),
        collate_fns=[train_collate_fn] * len(train_datasets),
    )

    _val_collate_fn = partial(
        val_collate_fn, use_external_attn_maps=config.use_external_attn_maps
    )
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(val_datasets),
        is_trains=[False] * len(val_datasets),
        collate_fns=[_val_collate_fn] * len(val_datasets),  # ✅ 修正写法
    )

    return train_loaders, val_loaders


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, val_loaders = setup_dataloaders(config)

    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = (
        num_steps_per_epoch * config.scheduler.warmup_epochs
    )
    torch.backends.cudnn.benchmark = True

    model_cls = eval(config.model.get("model_cls", "Chat3D"))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    save_step_interval = 1
    start_time = time.time()
    if not config.evaluate:
        logger.info("Start training")
        for epoch in range(start_epoch, config.scheduler.epochs):
            global_step = train(
                model,
                model_without_ddp,
                train_loaders,
                val_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                do_eval=config.do_eval,
            )
            if is_main_process():
                logger.info(f"Epoch {epoch}")
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                state_dict = model_without_ddp.state_dict()
                for k in list(state_dict.keys()):
                    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if (
                    (
                        (epoch + 1) % save_step_interval == 0
                        or epoch == config.scheduler.epochs - 1
                    )
                    and config.do_save
                    and not config.debug
                ):
                    if config.get("save_latest", False):
                        torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
                    else:
                        torch.save(
                            save_obj,
                            join(
                                config.output_dir, f"ckpt_{epoch:02d}_{global_step}.pth"
                            ),
                        )

            if global_step > max_global_step:
                break
            dist.barrier()

    if config.evaluate:
        evaluate_all(
            model,
            model_without_ddp,
            val_loaders,
            start_epoch - 1,
            global_step,
            device,
            config,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
