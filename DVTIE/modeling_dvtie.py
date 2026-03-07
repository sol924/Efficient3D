import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cmt_module import CMT, CMTConfig
from transformers import RobertaModel, RobertaTokenizer


@dataclass
class DVTIENetConfig:
    intput_feat_dim: int = 2048
    roberta_hidden_size: int = 768
    dropout: float = 0.3
    roberta_path: str = "/gpfs/work/cpt/yuhuilin21/PointLLM/roberta"
    mm_encoder: CMTConfig = CMTConfig()
    max_obj_num: int = 100
    train_text_encoder: bool = False
    enable_timing: bool = False


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size // 2),
        nn.ReLU(),
        nn.LayerNorm(hidden_size // 2, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, output_size),
    )


class DVTIENet(nn.Module):
    def __init__(self, config: DVTIENetConfig):
        super().__init__()
        self.config = config
        self.enable_timing = bool(getattr(config, "enable_timing", False))
        self._last_bert_ms = 0.0
        self._last_cmt_ms = 0.0

        # text encoder
        self.text_tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)
        self.text_encoder = RobertaModel.from_pretrained(config.roberta_path)

        if not config.train_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # mm encoder
        self.mm_encoder = CMT(config.mm_encoder)

        # projectors
        self.text_projector = nn.Linear(
            config.roberta_hidden_size, config.mm_encoder.hidden_size
        )
        self.obj_feat_projector = nn.Linear(
            config.intput_feat_dim, config.mm_encoder.hidden_size
        )
        # obj_id embedding
        self.obj_id_embedding = nn.Embedding(
            config.max_obj_num, config.mm_encoder.hidden_size
        )
        # output head
        self.out_head_attn = get_mlp_head(
            config.mm_encoder.hidden_size,
            config.mm_encoder.hidden_size,
            1,
            dropout=config.dropout,
        )

    def encode_text(self, prompt):
        text_tokens = self.text_tokenizer(prompt, return_tensors="pt", padding=True)
        text_tokens = {k: v.to(self.text_encoder.device) for k, v in text_tokens.items()}
        if self.enable_timing:
            device = self.text_encoder.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                txt_embeds = self.text_encoder(**text_tokens).last_hidden_state
                end_event.record()
                end_event.synchronize()
                self._last_bert_ms = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                txt_embeds = self.text_encoder(**text_tokens).last_hidden_state
                self._last_bert_ms = (time.perf_counter() - start_time) * 1000.0
        else:
            txt_embeds = self.text_encoder(**text_tokens).last_hidden_state
        txt_masks = text_tokens["attention_mask"]
        return txt_embeds, txt_masks

    def forward(
        self,
        object_features,
        object_img_features,
        object_locations,
        prompt,
        # mentioned_oids=None,
        attn_maps=None,
        **kwargs,
    ):
        # text encoding
        txt_embeds, txt_masks = self.encode_text(prompt)
        txt_embeds = self.text_projector(txt_embeds)
        # object encoding
        obj_embeds = torch.cat([object_features, object_img_features], dim=-1)
        obj_embeds = self.obj_feat_projector(obj_embeds)
        obj_masks = torch.ones(
            (obj_embeds.shape[0], obj_embeds.shape[1]),
            dtype=txt_masks.dtype,
            device=obj_embeds.device,
        )
        obj_embeds = obj_embeds + self.obj_id_embedding.weight
        # obj_embeds = obj_embeds
        # mm encoding
        if self.enable_timing:
            device = obj_embeds.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out_embeds = self.mm_encoder(
                    txt_embeds,
                    txt_masks,
                    obj_embeds,
                    object_locations,
                    obj_masks,
                )["obj_embeds"]
                end_event.record()
                end_event.synchronize()
                self._last_cmt_ms = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                out_embeds = self.mm_encoder(
                    txt_embeds,
                    txt_masks,
                    obj_embeds,
                    object_locations,
                    obj_masks,
                )["obj_embeds"]
                self._last_cmt_ms = (time.perf_counter() - start_time) * 1000.0
        else:
            out_embeds = self.mm_encoder(
                txt_embeds,
                txt_masks,
                obj_embeds,
                object_locations,
                obj_masks,
            )["obj_embeds"]
        # output
        out_logits = self.out_head_attn(out_embeds).squeeze(-1)
        ret = {"logits": out_logits, "loss": torch.tensor(0.0, device=out_logits.device)}
        if attn_maps is not None:
            kd_loss = distribution_matching_loss(out_logits, attn_maps)
            ret["loss"] += kd_loss
            ret.update(
                {
                    "kd_loss": kd_loss,
                }
            )
        if self.enable_timing:
            ret["timing"] = {
                "bert_ms": float(self._last_bert_ms),
                "cmt_ms": float(self._last_cmt_ms),
            }
        return ret

    def forward_with_tokens(
        self,
        object_features,
        object_img_features,
        object_locations,
        input_ids,
        attention_mask,
    ):
        """
        Forward path without tokenizer, for FLOPs tracing (tensor-only inputs).
        Returns logits tensor only.
        """
        txt_embeds = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        txt_masks = attention_mask
        txt_embeds = self.text_projector(txt_embeds)

        obj_embeds = torch.cat([object_features, object_img_features], dim=-1)
        obj_embeds = self.obj_feat_projector(obj_embeds)
        obj_masks = torch.ones(
            (obj_embeds.shape[0], obj_embeds.shape[1]),
            dtype=txt_masks.dtype,
            device=obj_embeds.device,
        )
        obj_embeds = obj_embeds + self.obj_id_embedding.weight
        out_embeds = self.mm_encoder(
            txt_embeds,
            txt_masks,
            obj_embeds,
            object_locations,
            obj_masks,
        )["obj_embeds"]
        out_logits = self.out_head_attn(out_embeds).squeeze(-1)
        return out_logits


def distribution_matching_loss(out_logits, attn_maps):
    """
    Computes a loss to match the distribution of out_logits to the attn_maps.

    Args:
        out_logits (torch.Tensor): Tensor of shape (B, N) representing model outputs.
        attn_maps (torch.Tensor): Tensor of shape (B, N) representing attention maps (not necessarily normalized).

    Returns:
        torch.Tensor: A scalar tensor representing the loss.
    """
    # Normalize attn_maps to create a probability distribution
    attn_maps_normalized = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)

    # Normalize out_logits to create a probability distribution
    out_logits_normalized = F.softmax(out_logits, dim=-1)

    # Compute the KL divergence loss (optionally, you can use MSE loss)
    loss = F.kl_div(
        torch.log(out_logits_normalized + 1e-8),  # log(P) for KL divergence
        attn_maps_normalized,  # Q is now normalized
        reduction="batchmean",  # Average over the batch
    )

    return loss
