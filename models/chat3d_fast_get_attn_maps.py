import random
import logging
from abc import ABC
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

# from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model

# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence

import contextlib
from dataset.base_dataset import update_caption, recover_caption

logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print(
            "{:80s}{:20s}{:20s}{}".format(
                name,
                "(Trainable)" if p.requires_grad else "(Fixed)",
                "(Has grad):" if p.grad is not None else "(No grad backward):",
                list(p.shape),
            )
        )


class Chat3D(nn.Module):
    """
    VideoChat model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.start_sym = config.model.start_sym
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj
        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num
        self.bidirection = config.model.bidirection
        self.add_pos_emb = config.model.add_pos_emb
        self.feat_fusion = config.model.feat_fusion
        self.fuse_with_id = config.model.fuse_with_id
        self.use_location_token = config.model.use_location_token
        self.token_pruning = getattr(config, "token_pruning", True)

        self.debug = config.debug
        if not self.debug:
            logger.info("Loading LLaMA")
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(
                llama_model_path, use_fast=False, legacy=False
            )
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    # attn_implementation="flash_attention_2",
                    use_cache=False,
                    output_attentions=True,
                )
                use_fast_v = bool(self.config.use_fast_v) and self.token_pruning
                use_fast_v_oracle = bool(self.config.use_fast_v_oracle) and self.token_pruning
                self.llama_model.config.use_fast_v = use_fast_v
                self.llama_model.config.fast_v_sys_length = 45
                self.llama_model.config.fast_v_image_token_length = 300
                self.llama_model.config.fast_v_attention_rank = (
                    self.config.fast_v_attention_rank
                )
                self.llama_model.config.fast_v_agg_layer = self.config.fast_v_agg_layer
                self.llama_model.config.use_fast_v_oracle = use_fast_v_oracle
                self.llama_model.model.reset_fastv()
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:

                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in ["instance2embed", "hidden_state2query"]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))

                lora_target_modules = find_linear_layers(
                    self.llama_model, config.lora.lora_target_modules
                )

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.lm_head.weight.requires_grad = True
                self.llama_model.model.lm_head.weight.data = (
                    self.llama_model.model.lm_head.weight.data.float()
                )
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = (
                    self.llama_model.model.model.embed_tokens.weight.data.float()
                )
                self.llama_model.print_trainable_parameters()
            else:
                self.llama_model.lm_head.weight.requires_grad = True
                self.llama_model.lm_head.weight.data = (
                    self.llama_model.lm_head.weight.data.float()
                )
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = (
                    self.llama_model.model.embed_tokens.weight.data.float()
                )

            # self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            objid_tokens = []
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.objid_end_idx = len(self.llama_tokenizer)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            self.llama_dim = self.llama_model.config.hidden_size
            logger.info("Loading LLAMA Done")
        else:
            self.llama_model = None
            self.llama_dim = 4096

        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim),
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim),
        )
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(nn.Linear(self.pos_dim, self.llama_dim))

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None

    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[
                self.objid_start_idx : self.objid_end_idx
            ]  # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[
                self.objid_start_idx : self.objid_end_idx
            ]
        return objid_embeds

    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": "
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(
            p_0, return_tensors="pt", add_special_tokens=True
        )
        p_1_token = self.llama_tokenizer(
            p_1, return_tensors="pt", add_special_tokens=False
        )
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat

    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs() ** dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_object_list_embed(
        self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids
    ):
        valid_ids = torch.where(scene_mask)[0].tolist()
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[
                self.objid_start_idx : self.objid_end_idx
            ]  # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[
                self.objid_start_idx : self.objid_end_idx
            ]

        assigned_ids = assigned_ids[valid_ids]
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]
        if self.use_location_token:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::2, :] += embed_obj[assigned_ids]
            object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.fuse_with_id:
            object_list_embed = selected_objid_embeds
            if not self.no_obj:
                object_list_embed += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed += embed_img[assigned_ids]
            return object_list_embed
        if self.feat_fusion:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::2, :] = selected_objid_embeds
            if not self.no_obj:
                object_list_embed[1::2, :] += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.no_obj:
            # if embed_img is None:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[assigned_ids]
            return object_list_embed
            # object_list_embed = selected_objid_embeds + embed_obj[assigned_ids]
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_scene[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros(
                (selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]),
                dtype=selected_objid_embeds.dtype,
                device=selected_objid_embeds.device,
            )
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[assigned_ids]
            object_list_embed[2::4, :] = embed_scene[assigned_ids]
            object_list_embed[3::4, :] = embed_img[assigned_ids]
            return object_list_embed
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float("inf")))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float("-inf")))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(
        self,
        scene_feat,
        scene_img_feat,
        scene_locs,
        scene_mask,
        obj_ids,
        assigned_ids,
        questions,
        answers,
        is_eval=False,
        **kwargs,
    ):
        object_embed, object_img_embed = self.encode_object_feat(
            scene_feat, scene_img_feat, scene_locs
        )
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = (
                self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            )
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(
                scene_embed, src_key_padding_mask=~scene_mask
            )
            proj_scene_embed = self.scene_proj(scene_embed)

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        for i, question in enumerate(questions):
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i],
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i],
            )
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            object_list_intervals.append(
                (p_0_embed.shape[0], p_0_embed.shape[0] + object_list_embed.shape[0])
            )
            wrapped_embed = torch.cat(
                [p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0
            )
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(
                wrapped_embed.device
            )
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(
                answer, return_tensors="pt", add_special_tokens=False
            ).to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            # to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])

        max_seq_len = min(768, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(
                tensor_list, batch_first=batch_first, padding_value=padding_value
            )
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded

        input_embeds = pad_and_trim(
            input_embed_list, max_seq_len, batch_first=True, padding_value=0
        ).to(device)
        targets = pad_and_trim(
            target_list, max_seq_len, batch_first=True, padding_value=-100
        ).to(device)
        attention_mask = pad_and_trim(
            attn_list, max_seq_len, batch_first=True, padding_value=0
        ).to(device)
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones(
                (max_seq_len, max_seq_len), dtype=input_dtype, device=device
            )
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = (
                causal_mask[None, None, :, :]
                .expand(input_embeds.shape[0], 1, -1, -1)
                .clone()
            )
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # label_weights=label_weights
            )

        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu()
            if proj_scene_embed is not None
            else 0.0,
            max_seq_len=max_seq_len,
        )

    # def get_attn_maps(
    #     self,
    #     scene_feat,
    #     scene_img_feat,
    #     scene_locs,
    #     scene_mask,
    #     prompts,
    #     captions,
    #     obj_ids,
    #     assigned_ids,
    #     **kwargs,
    # ):
    #     object_embed, object_img_embed = self.encode_object_feat(
    #         scene_feat, scene_img_feat, scene_locs
    #     )
    #     device = object_embed.device
    #     batch_size, obj_num = object_embed.shape[:2]
    #     proj_object_embed = self.object_proj(object_embed)
    #     proj_object_img_embed = self.object_img_proj(object_img_embed)
    #     if self.add_pos_emb:
    #         mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
    #         pos_embed = (
    #             self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
    #         )
    #         proj_pos_embed = self.pos_proj(pos_embed)
    #         proj_object_embed = proj_object_embed + proj_pos_embed
    #         proj_object_img_embed = proj_object_img_embed + proj_pos_embed
    #     if self.add_scene_token:
    #         # if self.add_img_token:
    #         #     object_embed = object_embed + object_img_embed
    #         obj_embed = self.scene_init_proj(object_embed)
    #         mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
    #         pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
    #         pos_embed = self.pos_proj(pos_embed)
    #         scene_embed = obj_embed + pos_embed
    #         scene_embed = self.relation_module(
    #             scene_embed, src_key_padding_mask=~scene_mask
    #         )
    #         proj_scene_embed = self.scene_proj(scene_embed)
    #
    #     output_texts = []
    #     p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
    #     p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
    #     batch_wrapped_embeds = []
    #     # batch_attention_masks = []
    #     ret_a_maps = []
    #     for i in range(batch_size):
    #         tmp_prompt = f" {prompts[i]} "
    #         tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
    #         prompt_embed = self.get_text_emb(tmp_prompt, device=device)
    #
    #         caption = captions[i]
    #         tmp_prompt2 = f" {prompts[i]} {self.role[1]}: {caption}"
    #         tmp_prompt2 = update_caption(tmp_prompt2, assigned_ids[i])
    #         prompt_embed2 = self.get_text_emb(tmp_prompt2, device=device)
    #
    #         object_list_embed = self.get_object_list_embed(
    #             proj_object_embed[i],
    #             proj_object_img_embed[i] if self.add_img_token else None,
    #             proj_scene_embed[i] if self.add_scene_token else None,
    #             scene_mask[i],
    #             obj_ids[i],
    #             assigned_ids[i],
    #         )
    #
    #         object_list_embed = object_list_embed.unsqueeze(0)
    #         wrapped_embed = torch.cat(
    #             [p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1
    #         )
    #         wrapped_embed2 = torch.cat(
    #             [p_0_embed, object_list_embed, p_1_embed, prompt_embed2], dim=1
    #         )
    #         attention_mask = None
    #         if self.bidirection:
    #             seq_len = wrapped_embed.shape[1]
    #             attention_mask = torch.ones(
    #                 (seq_len, seq_len), dtype=wrapped_embed.dtype, device=device
    #             )
    #             attention_mask = torch.tril(attention_mask, diagonal=0)
    #             attention_mask = (
    #                 attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
    #             )
    #             st, ed = (
    #                 p_0_embed.shape[1],
    #                 p_0_embed.shape[1] + object_list_embed.shape[1],
    #             )
    #             attention_mask[:, :, st:ed, st:ed] = 1.0
    #         # debug
    #         batch_wrapped_embeds.append(wrapped_embed.squeeze(0))
    #
    #         self.llama_model.config.use_fast_v = False
    #         self.llama_model.config.use_fast_v_oracle = False
    #         self.llama_model.base_model.model.model.reset_fastv()
    #         with self.maybe_autocast():
    #             outputs = self.llama_model.generate(
    #                 inputs_embeds=wrapped_embed2,
    #                 max_new_tokens=self.max_txt_len,
    #                 num_beams=1,
    #                 min_length=1,
    #                 repetition_penalty=3.0,
    #                 length_penalty=1,
    #                 temperature=1.0,
    #                 use_cache=False,
    #                 output_attentions=True,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #             )
    #         # a_map = torch.stack(
    #         #     [outputs.attentions[0][i][0].mean(0) for i in range(32)]
    #         # ).mean(0)
    #         st0 = p_0_embed.shape[1] + object_list_embed.shape[1] + p_1_embed.shape[1]
    #         st1, ed1 = p_0_embed.shape[1], p_0_embed.shape[1] + object_list_embed.shape[1]
    #         # a_map = a_map[st0:, st1:ed1]
    #         # a_map = a_map.reshape(-1, 100, 3).sum(-1)
    #         # a_map_ori = a_map[: prompt_embed.shape[1]].mean(0)
    #         # a_map = a_map.mean(0)
    #         # 这里不要 mean(0)，保留 32 个 head
    #         a_map = torch.stack(
    #             [outputs.attentions[0][i][0].mean(0) for i in range(32)]
    #         )  # -> [32, seq_len, seq_len]
    #         a_map = a_map[:, st0:, st1:ed1]  # -> [32, T_text, T_obj]
    #         a_map = a_map.reshape(32, -1, 100, 3).sum(-1)  # -> [32, T_text, 100]
    #         a_map_ori = a_map[:, :prompt_embed.shape[1]].mean(1)  # -> [32, 100]
    #         a_map = a_map.mean(1)  # -> [32, 100]
    #         print(a_map.shape, a_map_ori.shape)
    #         ret_a_maps.append(
    #             {
    #                 "index": kwargs["index"][i],
    #                 "a_map": a_map.cpu(),
    #                 "a_map_ori": a_map_ori.cpu(),
    #             }
    #         )
    #
    #     return ret_a_maps

    def get_attn_maps(
            self,
            scene_feat,
            scene_img_feat,
            scene_locs,
            scene_mask,
            prompts,
            captions,
            obj_ids,
            assigned_ids,
            **kwargs,
    ):
        """
        返回 ret_a_maps: list[dict]，每个样本一个 dict。
        每个字段都是 [batch, layer, 100]：
            batch: 这次 generate 的 batch 大小（通常为1，因为我们逐样本调用）
            layer: LLaMA 层数（不做layer平均，逐层保留）
            100:   最多100个物体

        字段含义（逐层）：
            a_map                prompt+caption+生成 的 text→object 注意力 (≈ a_prompt + a_text)
            a_map_ori            仅 prompt 的 text→object 注意力 (≈ a_prompt)
            a_self               object→object 注意力 (≈ a_self)
            a_text_only          仅生成部分 text token 的注意力 (≈ a_text)
            a_self_plus_prompt   a_self + a_prompt
            a_self_plus_text     a_self + a_text
            a_self_plus_all      a_self + a_prompt + a_text
        """

        # ===== 1. 预处理特征到 LLaMA 空间 =====
        object_embed, object_img_embed = self.encode_object_feat(
            scene_feat, scene_img_feat, scene_locs
        )
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]

        proj_object_embed = self.object_proj(object_embed)  # [B, N_obj, D_llm]
        proj_object_img_embed = self.object_img_proj(object_img_embed)  # [B, N_obj, D_llm]

        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = (
                    self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            )
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        if self.add_scene_token:
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(
                scene_embed, src_key_padding_mask=~scene_mask
            )
            proj_scene_embed = self.scene_proj(scene_embed)

        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)  # [1, T_p0, D_llm]
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)  # [1, T_p1, D_llm]

        batch_wrapped_embeds = []
        ret_a_maps = []

        # ===== 2. 遍历 batch 的每个样本（我们按样本单独generate） =====
        for i in range(batch_size):
            # ---- 文本 prompt ----
            tmp_prompt = f" {prompts[i]} "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            # [1, T_prompt, D_llm] 纯指令，不带caption

            caption = captions[i]
            tmp_prompt2 = f" {prompts[i]} {self.role[1]}: {caption}"
            tmp_prompt2 = update_caption(tmp_prompt2, assigned_ids[i])
            prompt_embed2 = self.get_text_emb(tmp_prompt2, device=device)
            # [1, T_prompt2, D_llm] 指令+caption，将作为generate起始

            # ---- 视觉对象 tokens ----
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i],
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i],
            )  # [T_obj, D_llm]

            object_list_embed = object_list_embed.unsqueeze(0)  # [1, T_obj, D_llm]

            # ---- 拼接到 LLaMA 输入 ----
            wrapped_embed = torch.cat(
                [p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1
            )  # prompt-only 版本
            wrapped_embed2 = torch.cat(
                [p_0_embed, object_list_embed, p_1_embed, prompt_embed2], dim=1
            )  # prompt+caption 版本，拿去 generate

            attention_mask = None
            if self.bidirection:
                seq_len = wrapped_embed.shape[1]
                attention_mask = torch.ones(
                    (seq_len, seq_len), dtype=wrapped_embed.dtype, device=device
                )
                attention_mask = torch.tril(attention_mask, diagonal=0)
                attention_mask = attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
                st_tmp = p_0_embed.shape[1]
                ed_tmp = p_0_embed.shape[1] + object_list_embed.shape[1]
                attention_mask[:, :, st_tmp:ed_tmp, st_tmp:ed_tmp] = 1.0

            batch_wrapped_embeds.append(wrapped_embed.squeeze(0))

            # ===== 3. LLaMA generate，拿注意力 =====
            self.llama_model.config.use_fast_v = False
            self.llama_model.config.use_fast_v_oracle = False
            self.llama_model.base_model.model.model.reset_fastv()

            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed2,
                    max_new_tokens=self.max_txt_len,
                    num_beams=1,
                    min_length=1,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                    use_cache=False,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # 现在 outputs.attentions 的结构（decoder-only模型）通常是：
            # outputs.attentions[step_idx][layer_idx] -> [batch, num_heads, seq_len, seq_len]
            # 我们只看 step_idx = 0（第一步的注意力，有完整上下文）
            attentions_step0 = outputs.attentions[0]  # tuple length = num_layers

            num_layers = len(attentions_step0)
            # 从任意一层拿到 batch size
            local_bs = attentions_step0[0].shape[0]

            # ---- 序列切分位置 ----
            obj_st = p_0_embed.shape[1]
            obj_ed = p_0_embed.shape[1] + object_list_embed.shape[1]

            txt_st = p_0_embed.shape[1] + object_list_embed.shape[1] + p_1_embed.shape[1]
            # 从 txt_st 到序列末尾是：prompt_embed2（指令+caption）以及后续生成token

            pure_prompt_len = prompt_embed.shape[1]  # 只有原始prompt，不含caption

            # ===== 4. 为每一层单独算7个 supervision 向量 =====
            # 我们会得到 list[num_layers]，其中每个元素形状是 [local_bs, 100]
            a_map_list = []
            a_map_ori_list = []
            a_self_list = []
            a_text_only_list = []
            a_self_plus_prompt_list = []
            a_self_plus_text_list = []
            a_self_plus_all_list = []

            for layer_idx in range(num_layers):
                # attentions_step0[layer_idx]: [local_bs, num_heads, seq_len, seq_len]
                layer_attn = attentions_step0[layer_idx]

                # 对 head 做平均 (保持 batch 维) -> [local_bs, seq_len, seq_len]
                A_layer = layer_attn.mean(dim=1)

                # ---------- text -> object ----------
                # query: 文本 tokens [txt_st:, :]
                # key:   物体 tokens [obj_st:obj_ed]
                # -> [local_bs, T_text, T_obj]
                A_text_to_obj_layer = A_layer[:, txt_st:, obj_st:obj_ed]

                # 把每个object的3个sub-token合并：
                # reshape -> [local_bs, T_text, 100, 3] -> sum(-1) -> [local_bs, T_text, 100]
                A_text_to_obj_layer = A_text_to_obj_layer.reshape(local_bs, -1, 100, 3).sum(-1)

                # a_prompt_layer: 仅prompt tokens的平均 => [local_bs, 100]
                a_prompt_layer = A_text_to_obj_layer[:, :pure_prompt_len].mean(dim=1)

                # a_prompt_all_layer: 所有文本token (prompt+caption+生成) 平均 => [local_bs, 100]
                a_prompt_all_layer = A_text_to_obj_layer.mean(dim=1)
                # 这是 prompt + text 的混合

                # a_text_layer: 生成token额外贡献 => [local_bs, 100]
                a_text_layer = (a_prompt_all_layer - a_prompt_layer).clamp(min=0.0)

                # ---------- object -> object ----------
                # query: object tokens
                # key:   object tokens
                # -> [local_bs, T_obj, T_obj]
                A_obj_obj_layer = A_layer[:, obj_st:obj_ed, obj_st:obj_ed]

                # “被大家看了多少”：对 query 维 (dim=1) 做平均 => [local_bs, T_obj]
                obj_token_importance_layer = A_obj_obj_layer.mean(dim=1)

                # 合并3个sub-token => [local_bs, 100]
                a_self_layer = obj_token_importance_layer.reshape(local_bs, 100, 3).sum(dim=-1)

                # ---------- 组合 ----------
                # self + prompt
                a_self_plus_prompt_layer = a_self_layer + a_prompt_layer  # [local_bs, 100]
                # self + text
                a_self_plus_text_layer = a_self_layer + a_text_layer  # [local_bs, 100]
                # self + prompt + text
                a_self_plus_all_layer = a_self_layer + a_prompt_layer + a_text_layer  # [local_bs, 100]

                # ---------- 累积 ----------
                a_map_list.append(a_prompt_all_layer)  # prompt+caption+生成
                a_map_ori_list.append(a_prompt_layer)  # prompt only
                a_self_list.append(a_self_layer)  # object self-attn
                a_text_only_list.append(a_text_layer)  # text only
                a_self_plus_prompt_list.append(a_self_plus_prompt_layer)
                a_self_plus_text_list.append(a_self_plus_text_layer)
                a_self_plus_all_list.append(a_self_plus_all_layer)

            # ===== 5. 把 [num_layers * (local_bs,100)] 叠成 [local_bs, num_layers, 100] =====
            # 现在每个 list 里有 num_layers 个 [local_bs, 100]，
            # stack(dim=1) 后就是 [local_bs, num_layers, 100]，完全符合 (batch, layer, 100)
            a_map_tensor = torch.stack(a_map_list, dim=1)  # [local_bs, L, 100]
            a_map_ori_tensor = torch.stack(a_map_ori_list, dim=1)  # [local_bs, L, 100]
            a_self_tensor = torch.stack(a_self_list, dim=1)  # [local_bs, L, 100]
            a_text_only_tensor = torch.stack(a_text_only_list, dim=1)  # [local_bs, L, 100]
            a_self_plus_prompt_tensor = torch.stack(a_self_plus_prompt_list, dim=1)  # [local_bs, L, 100]
            a_self_plus_text_tensor = torch.stack(a_self_plus_text_list, dim=1)  # [local_bs, L, 100]
            a_self_plus_all_tensor = torch.stack(a_self_plus_all_list, dim=1)  # [local_bs, L, 100]

            # # ===== 6. 打印形状确认 =====
            # # 你应该会看到形状类似 torch.Size([1, 32, 100])
            # # 如果还是 torch.Size([1, 1, 100])，那说明 generate 目前只返回了1层注意力
            # print("a_map shape:", a_map_tensor.shape)
            # print("a_map_ori shape:", a_map_ori_tensor.shape)
            # print("a_self shape:", a_self_tensor.shape)
            # print("a_text_only shape:", a_text_only_tensor.shape)
            # print("a_self_plus_prompt shape:", a_self_plus_prompt_tensor.shape)
            # print("a_self_plus_text shape:", a_self_plus_text_tensor.shape)
            # print("a_self_plus_all shape:", a_self_plus_all_tensor.shape)

            # ===== 7. 收集到 ret_a_maps =====
            ret_a_maps.append(
                {
                    "index": kwargs["index"][i],

                    # 每个张量都是 [local_bs, num_layers, 100]
                    "a_map": a_map_tensor.detach().cpu(),
                    "a_map_ori": a_map_ori_tensor.detach().cpu(),
                    "a_self": a_self_tensor.detach().cpu(),
                    "a_text_only": a_text_only_tensor.detach().cpu(),
                    "a_self_plus_prompt": a_self_plus_prompt_tensor.detach().cpu(),
                    "a_self_plus_text": a_self_plus_text_tensor.detach().cpu(),
                    "a_self_plus_all": a_self_plus_all_tensor.detach().cpu(),
                }
            )

        return ret_a_maps

    def forward(self, **kwargs):
        return self.get_attn_maps(**kwargs)

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
