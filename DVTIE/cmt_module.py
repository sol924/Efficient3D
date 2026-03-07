# import copy
# import numpy as np
# from typing import Optional
# import time
#
# import einops
#
# import torch
# from torch import nn, Tensor
# import torch.nn.functional as F
# from dataclasses import dataclass
#
#
# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
#
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
#
# class TransformerDecoderLayer(nn.Module):
#     def __init__(
#         self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
#     ):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             d_model, nhead, dropout=dropout, batch_first=True
#         )
#         self.multihead_attn = nn.MultiheadAttention(
#             d_model, nhead, dropout=dropout, batch_first=True
#         )
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def forward(
#         self,
#         tgt,
#         memory,
#         tgt_mask: Optional[Tensor] = None,
#         memory_mask: Optional[Tensor] = None,
#         tgt_key_padding_mask: Optional[Tensor] = None,
#         memory_key_padding_mask: Optional[Tensor] = None,
#     ):
#         tgt2 = self.norm1(tgt)
#         tgt2, self_attn_matrices = self.self_attn(
#             tgt2,
#             tgt2,
#             value=tgt2,
#             attn_mask=tgt_mask,
#             key_padding_mask=tgt_key_padding_mask,
#         )
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2, cross_attn_matrices = self.multihead_attn(
#             query=tgt2,
#             key=memory,
#             value=memory,
#             attn_mask=memory_mask,
#             key_padding_mask=memory_key_padding_mask,
#         )
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt, self_attn_matrices, cross_attn_matrices
#
#
# class MultiHeadAttentionSpatial(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         n_head,
#         dropout=0.1,
#         spatial_multihead=True,
#         spatial_dim=5,
#         spatial_attn_fusion="mul",
#     ):
#         super().__init__()
#         assert d_model % n_head == 0, "d_model: %d, n_head: %d" % (d_model, n_head)
#
#         self.n_head = n_head
#         self.d_model = d_model
#         self.d_per_head = d_model // n_head
#         self.spatial_multihead = spatial_multihead
#         self.spatial_dim = spatial_dim
#         self.spatial_attn_fusion = spatial_attn_fusion
#
#         self.w_qs = nn.Linear(d_model, d_model)
#         self.w_ks = nn.Linear(d_model, d_model)
#         self.w_vs = nn.Linear(d_model, d_model)
#
#         self.fc = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(p=dropout)
#         self.layer_norm = nn.LayerNorm(d_model)
#
#         self.spatial_n_head = n_head if spatial_multihead else 1
#         if self.spatial_attn_fusion in ["mul", "bias", "add"]:
#             self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
#         elif self.spatial_attn_fusion == "ctx":
#             self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
#         elif self.spatial_attn_fusion == "cond":
#             self.lang_cond_fc = nn.Linear(
#                 d_model, self.spatial_n_head * (spatial_dim + 1)
#             )
#         else:
#             raise NotImplementedError(
#                 "unsupported spatial_attn_fusion %s" % (self.spatial_attn_fusion)
#             )
#
#     def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
#         residual = q
#         q = einops.rearrange(self.w_qs(q), "b l (head k) -> head b l k", head=self.n_head)
#         k = einops.rearrange(self.w_ks(k), "b t (head k) -> head b t k", head=self.n_head)
#         v = einops.rearrange(self.w_vs(v), "b t (head v) -> head b t v", head=self.n_head)
#         attn = torch.einsum("hblk,hbtk->hblt", q, k) / np.sqrt(q.shape[-1])
#
#         if self.spatial_attn_fusion in ["mul", "bias", "add"]:
#             loc_attn = self.pairwise_loc_fc(pairwise_locs)
#             loc_attn = einops.rearrange(loc_attn, "b l t h -> h b l t")
#             if self.spatial_attn_fusion == "mul":
#                 loc_attn = F.relu(loc_attn)
#             if not self.spatial_multihead:
#                 loc_attn = einops.repeat(
#                     loc_attn, "h b l t -> (h nh) b l t", nh=self.n_head
#                 )
#         elif self.spatial_attn_fusion == "ctx":
#             loc_attn = self.pairwise_loc_fc(pairwise_locs)
#             loc_attn = einops.rearrange(
#                 loc_attn, "b l t (h k) -> h b l t k", h=self.n_head
#             )
#             loc_attn = torch.einsum("hblk,hbltk->hblt", q, loc_attn) / np.sqrt(
#                 q.shape[-1]
#             )
#         elif self.spatial_attn_fusion == "cond":
#             spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
#             spatial_weights = einops.rearrange(
#                 spatial_weights,
#                 "b l (h d) -> h b l d",
#                 h=self.spatial_n_head,
#                 d=self.spatial_dim + 1,
#             )
#             if self.spatial_n_head == 1:
#                 spatial_weights = einops.repeat(
#                     spatial_weights, "1 b l d -> h b l d", h=self.n_head
#                 )
#             spatial_bias = spatial_weights[..., :1]
#             spatial_weights = spatial_weights[..., 1:]
#             loc_attn = (
#                 torch.einsum("hbld,bltd->hblt", spatial_weights, pairwise_locs)
#                 + spatial_bias
#             )
#             loc_attn = torch.sigmoid(loc_attn)
#
#         if key_padding_mask is not None:
#             mask = einops.repeat(
#                 key_padding_mask, "b t -> h b l t", h=self.n_head, l=q.size(2)
#             )
#             attn = attn.masked_fill(mask, -np.inf)
#             if self.spatial_attn_fusion in ["mul", "cond"]:
#                 loc_attn = loc_attn.masked_fill(mask, 0)
#             else:
#                 loc_attn = loc_attn.masked_fill(mask, -np.inf)
#
#         if self.spatial_attn_fusion == "add":
#             fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
#         else:
#             if self.spatial_attn_fusion in ["mul", "cond"]:
#                 fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
#             else:
#                 fused_attn = loc_attn + attn
#             fused_attn = torch.softmax(fused_attn, 3)
#
#         assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)
#
#         output = torch.einsum("hblt,hbtv->hblv", fused_attn, v)
#         output = einops.rearrange(output, "head b l v -> b l (head v)")
#         output = self.dropout(self.fc(output))
#         output = self.layer_norm(output + residual)
#         return output, fused_attn
#
#
# class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
#     def __init__(
#         self,
#         d_model,
#         nhead,
#         dim_feedforward=2048,
#         dropout=0.1,
#         activation="relu",
#         spatial_multihead=True,
#         spatial_dim=5,
#         spatial_attn_fusion="mul",
#     ):
#         super().__init__(
#             d_model,
#             nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#         )
#         del self.self_attn
#         self.self_attn = MultiHeadAttentionSpatial(
#             d_model,
#             nhead,
#             dropout=dropout,
#             spatial_multihead=spatial_multihead,
#             spatial_dim=spatial_dim,
#             spatial_attn_fusion=spatial_attn_fusion,
#         )
#
#     def forward(
#         self,
#         tgt,
#         memory,
#         tgt_pairwise_locs,
#         tgt_mask: Optional[Tensor] = None,
#         memory_mask: Optional[Tensor] = None,
#         tgt_key_padding_mask: Optional[Tensor] = None,
#         memory_key_padding_mask: Optional[Tensor] = None,
#     ):
#         tgt2 = self.norm1(tgt)
#         tgt2, self_attn_matrices = self.self_attn(
#             tgt2,
#             tgt2,
#             tgt2,
#             tgt_pairwise_locs,
#             key_padding_mask=tgt_key_padding_mask,
#             txt_embeds=memory[:, 0],
#         )
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2, cross_attn_matrices = self.multihead_attn(
#             query=tgt2,
#             key=memory,
#             value=memory,
#             attn_mask=memory_mask,
#             key_padding_mask=memory_key_padding_mask,
#         )
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt, self_attn_matrices, cross_attn_matrices
#
#
# @dataclass
# class CMTConfig:
#     spatial_dec: bool = True
#     spatial_multihead: bool = True
#     spatial_dim: int = 5
#     spatial_dist_norm: bool = True
#     spatial_attn_fusion: str = "cond"
#     num_layers: int = 5
#     obj_loc_encoding: str = "same_all"
#     pairwise_rel_type: str = "center"
#     hidden_size: int = 768
#     num_attention_heads: int = 12
#     dim_loc: int = 6
#
#
# class CMT(nn.Module):
#     def __init__(self, config: CMTConfig):
#         super().__init__()
#         self.config = config
#
#         if self.config.spatial_dec:
#             decoder_class = TransformerSpatialDecoderLayer
#             kwargs = {
#                 "spatial_dim": config.spatial_dim,
#                 "spatial_multihead": config.spatial_multihead,
#                 "spatial_attn_fusion": config.spatial_attn_fusion,
#             }
#         else:
#             decoder_class = TransformerDecoderLayer
#             kwargs = {}
#
#         decoder_layer = decoder_class(
#             config.hidden_size,
#             config.num_attention_heads,
#             dim_feedforward=2048,
#             dropout=0.1,
#             activation="gelu",
#             **kwargs,
#         )
#         self.layers = _get_clones(decoder_layer, config.num_layers)
#
#         loc_layer = nn.Sequential(
#             nn.Linear(config.dim_loc, config.hidden_size),
#             nn.LayerNorm(config.hidden_size),
#         )
#         if self.config.obj_loc_encoding in ["same_0", "same_all"]:
#             num_loc_layers = 1
#         elif self.config.obj_loc_encoding == "diff_all":
#             num_loc_layers = config.num_layers
#         self.loc_layers = _get_clones(loc_layer, num_loc_layers)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """Initialize the weights"""
#         if isinstance(module, nn.Linear):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def calc_pairwise_locs(
#         self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type="center"
#     ):
#         if pairwise_rel_type == "mlp":
#             obj_locs = torch.cat([obj_centers, obj_whls], 2)
#             pairwise_locs = torch.cat(
#                 [
#                     einops.repeat(obj_locs, "b l d -> b l x d", x=obj_locs.size(1)),
#                     einops.repeat(obj_locs, "b l d -> b x l d", x=obj_locs.size(1)),
#                 ],
#                 dim=3,
#             )
#             return pairwise_locs
#
#         pairwise_locs = einops.repeat(obj_centers, "b l d -> b l 1 d") - einops.repeat(
#             obj_centers, "b l d -> b 1 l d"
#         )
#         pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps)  # (b, l, l)
#         if self.config.spatial_dist_norm:
#             max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[
#                 0
#             ]
#             norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, "b -> b 1 1")
#         else:
#             norm_pairwise_dists = pairwise_dists
#
#         if self.config.spatial_dim == 1:
#             return norm_pairwise_dists.unsqueeze(3)
#
#         pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
#         if pairwise_rel_type == "center":
#             pairwise_locs = torch.stack(
#                 [
#                     norm_pairwise_dists,
#                     pairwise_locs[..., 2] / pairwise_dists,
#                     pairwise_dists_2d / pairwise_dists,
#                     pairwise_locs[..., 1] / pairwise_dists_2d,
#                     pairwise_locs[..., 0] / pairwise_dists_2d,
#                 ],
#                 dim=3,
#             )
#         elif pairwise_rel_type == "vertical_bottom":
#             bottom_centers = torch.clone(obj_centers)
#             bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
#             bottom_pairwise_locs = einops.repeat(
#                 bottom_centers, "b l d -> b l 1 d"
#             ) - einops.repeat(bottom_centers, "b l d -> b 1 l d")
#             bottom_pairwise_dists = torch.sqrt(
#                 torch.sum(bottom_pairwise_locs**2, 3) + eps
#             )  # (b, l, l)
#             bottom_pairwise_dists_2d = torch.sqrt(
#                 torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps
#             )
#             pairwise_locs = torch.stack(
#                 [
#                     norm_pairwise_dists,
#                     bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
#                     bottom_pairwise_dists_2d / bottom_pairwise_dists,
#                     pairwise_locs[..., 1] / pairwise_dists_2d,
#                     pairwise_locs[..., 0] / pairwise_dists_2d,
#                 ],
#                 dim=3,
#             )
#
#         if self.config.spatial_dim == 4:
#             pairwise_locs = pairwise_locs[..., 1:]
#         return pairwise_locs
#
#     def forward(
#         self,
#         txt_embeds,
#         txt_masks,
#         obj_embeds,
#         obj_locs,
#         obj_masks,
#         output_attentions=False,
#         output_hidden_states=False,
#     ):
#         if self.config.spatial_dec:
#             pairwise_locs = self.calc_pairwise_locs(
#                 obj_locs[:, :, :3],
#                 obj_locs[:, :, 3:],
#                 pairwise_rel_type=self.config.pairwise_rel_type,
#             )
#
#         out_embeds = obj_embeds
#         all_hidden_states = [out_embeds]
#         all_self_attn_matrices, all_cross_attn_matrices = [], []
#         for i, layer in enumerate(self.layers):
#             if self.config.obj_loc_encoding == "diff_all":
#                 query_pos = self.loc_layers[i](obj_locs)
#                 out_embeds = out_embeds + query_pos
#             else:
#                 query_pos = self.loc_layers[0](obj_locs)
#                 if self.config.obj_loc_encoding == "same_all":
#                     out_embeds = out_embeds + query_pos
#                 else:
#                     if i == 0:
#                         out_embeds = out_embeds + query_pos
#
#             if self.config.spatial_dec:
#                 out_embeds, self_attn_matrices, cross_attn_matrices = layer(
#                     out_embeds,
#                     txt_embeds,
#                     pairwise_locs,
#                     tgt_key_padding_mask=obj_masks.logical_not(),
#                     memory_key_padding_mask=txt_masks.logical_not(),
#                 )
#             else:
#                 out_embeds, self_attn_matrices, cross_attn_matrices = layer(
#                     out_embeds,
#                     txt_embeds,
#                     tgt_key_padding_mask=obj_masks.logical_not(),
#                     memory_key_padding_mask=txt_masks.logical_not(),
#                 )
#
#             all_hidden_states.append(out_embeds)
#             all_self_attn_matrices.append(self_attn_matrices)
#             all_cross_attn_matrices.append(cross_attn_matrices)
#
#         outs = {
#             "obj_embeds": out_embeds,
#         }
#         if output_hidden_states:
#             outs["all_hidden_states"] = all_hidden_states
#         if output_attentions:
#             outs["all_self_attns"] = all_self_attn_matrices
#             outs["all_cross_attns"] = all_cross_attn_matrices
#         return outs
import copy
import numpy as np
from typing import Optional
import einops

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# =========================================
# Learnable Attention Mask for Cross-Attn (DHCT)
# =========================================
class DHCT(nn.Module):
    """
    learnable mask for cross-attention:
        M = sigmoid( (Q W_q) (K W_k)^T / (sqrt(d_dhct) * temp) ),  M in (0,1)
    """
    def __init__(self, d_model: int, d_dhct: int = 128, temp: float = 1.0, stop_grad_inputs: bool = False):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_dhct)
        self.proj_k = nn.Linear(d_model, d_dhct)
        self.scale = d_dhct ** 0.5
        self.temp = float(temp)
        self.stop_grad_inputs = bool(stop_grad_inputs)

        nn.init.normal_(self.proj_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj_k.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_q.bias)
        nn.init.zeros_(self.proj_k.bias)

    @torch.no_grad()
    def _apply_key_padding(self, M: torch.Tensor, key_padding_mask: Optional[torch.Tensor]):
        if key_padding_mask is None:
            return M
        # True 琛ㄧず pad锛氬皢 M 缃?0
        return M.masked_fill(key_padding_mask[:, None, :], 0.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q: [B, L_q, C], k: [B, L_k, C] -> M: [B, L_q, L_k] in (0,1)
        """
        if self.stop_grad_inputs:
            q = q.detach()
            k = k.detach()
        q_ = self.proj_q(q)  # [B,Lq,d_dhct]
        k_ = self.proj_k(k)  # [B,Lk,d_dhct]
        scores = torch.einsum('bqd,bkd->bqk', q_, k_) / (self.scale * self.temp)
        M = torch.sigmoid(scores)
        M = self._apply_key_padding(M, key_padding_mask)
        return M


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
        # DHCT 鍩烘湰椤?        use_dhct: bool = True, dhct_dim: int = 128, dhct_strength: float = 0.4,
        dhct_temp: float = 1.2, dhct_stop_grad_inputs: bool = False,
        # 杞诲井寮曞鐩稿叧
        dhct_use_log: bool = False, dhct_center_rows: bool = False,
        dhct_gate_bias: float = -6.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        # ---- DHCT (cross-attention only) ----
        self.use_dhct = use_dhct
        self.nhead = nhead
        self.dhct_strength = float(dhct_strength)
        self.dhct_use_log = bool(dhct_use_log)
        self.dhct_center_rows = bool(dhct_center_rows)
        if self.use_dhct:
            self.cross_dhct = DHCT(
                d_model, d_dhct=dhct_dim, temp=dhct_temp, stop_grad_inputs=dhct_stop_grad_inputs
            )

        # head-wise gate锛氬垵鍊煎嚑涔?0锛坰igmoid(-6)鈮?.0025锛?        self.dhct_head_gate = nn.Parameter(torch.zeros(self.nhead))
        self._dhct_gate_bias = float(dhct_gate_bias)

    def _build_dhct_bias(
        self,
        q: torch.Tensor,              # [B, L_q, C]
        k: torch.Tensor,              # [B, L_k, C]
        memory_key_padding_mask: Optional[Tensor],  # [B, L_k] or None
        base_attn_mask: Optional[Tensor]            # None or [..., L_q, L_k]
    ) -> Optional[Tensor]:
        """
        杩斿洖 additive mask锛坒loat锛? [B*H, L_q, L_k]
        缁熶竴灏?padding / DHCT / base mask 铻嶅悎涓轰竴涓姞鎬ф帺鐮侊紙涓嶅啀鍗曠嫭浼?key_padding_mask锛?        """
        B, Lq, Lk = q.size(0), q.size(1), k.size(1)

        # 1) base mask -> [B*H, Lq, Lk]
        if base_attn_mask is None:
            base = k.new_zeros((B * self.nhead, Lq, Lk), dtype=k.dtype, device=k.device)
        else:
            if base_attn_mask.dim() == 2:
                base = base_attn_mask.unsqueeze(0).expand(B * self.nhead, -1, -1).contiguous()
            elif base_attn_mask.dim() == 3:
                base = base_attn_mask
            else:
                raise ValueError("base_attn_mask must be 2D or 3D")
            base = base.to(dtype=k.dtype, device=k.device)

        # 2) padding -> additive -inf
        if memory_key_padding_mask is not None:
            pad_mask = memory_key_padding_mask.to(device=k.device)  # [B,Lk], bool
            pad_bias = torch.zeros((B, self.nhead, Lq, Lk), dtype=k.dtype, device=k.device)
            pad_bias = pad_bias.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            pad_bias = pad_bias.view(B * self.nhead, Lq, Lk)
            base = base + pad_bias

        # 3) DHCT
        if self.use_dhct:
            M = self.cross_dhct(q, k, key_padding_mask=memory_key_padding_mask)  # [B,Lq,Lk]

            if self.dhct_use_log:
                # 鐩稿杈冨己鐨勫紩瀵硷細log(M)锛堝彲閫夎涓績鍖栵級
                eps = 1e-5 if q.is_floating_point() else 1e-6
                dhct_term = torch.log(torch.clamp(M, min=eps))                   # [B,Lq,Lk]
                if self.dhct_center_rows:
                    dhct_term = dhct_term - dhct_term.mean(dim=-1, keepdim=True)
            else:
                # 寮卞紩瀵硷細鐩存帴鐢?(M-0.5)锛岃寖鍥?[-0.5,0.5]
                dhct_term = (M - 0.5)

            dhct_bias = self.dhct_strength * dhct_term                             # [B,Lq,Lk]
            dhct_bias = dhct_bias.unsqueeze(1).expand(B, self.nhead, Lq, Lk)      # [B,H,Lq,Lk]

            # head-wise gate锛?~1锛夛紝鍒濆€煎嚑涔?
            g = torch.sigmoid(self.dhct_head_gate + self._dhct_gate_bias).view(1, self.nhead, 1, 1)
            dhct_bias = dhct_bias * g
            dhct_bias = dhct_bias.reshape(B * self.nhead, Lq, Lk).to(dtype=k.dtype, device=k.device)
            base = base + dhct_bias

        return base  # [B*H, Lq, Lk]

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True, average_attn_weights=False
        )
        tgt = tgt + self.dropout1(tgt2)

        # Cross-Attention + DHCT bias
        tgt2 = self.norm2(tgt)
        fused_attn_mask = self._build_dhct_bias(
            q=tgt2, k=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            base_attn_mask=memory_mask,
        )
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory, value=memory,
            attn_mask=fused_attn_mask,
            key_padding_mask=None,  # 閬垮厤绫诲瀷鍐茬獊锛宲adding 宸插苟鍏?additive mask
            need_weights=True, average_attn_weights=False
        )
        tgt = tgt + self.dropout2(tgt2)

        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        dropout=0.1,
        spatial_multihead=True,
        spatial_dim=5,
        spatial_attn_fusion="mul",
    ):
        super().__init__()
        assert d_model % n_head == 0, f"d_model: {d_model}, n_head: {n_head}"
        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ["mul", "bias", "add"]:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == "ctx":
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == "cond":
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError(f"unsupported spatial_attn_fusion {self.spatial_attn_fusion}")

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), "b l (h k) -> h b l k", h=self.n_head)
        k = einops.rearrange(self.w_ks(k), "b t (h k) -> h b t k", h=self.n_head)
        v = einops.rearrange(self.w_vs(v), "b t (h v) -> h b t v", h=self.n_head)
        attn = torch.einsum("hblk,hbtk->hblt", q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ["mul", "bias", "add"]:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, "b l t h -> h b l t")
            if self.spatial_attn_fusion == "mul":
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, "h b l t -> (h nh) b l t", nh=self.n_head)
        elif self.spatial_attn_fusion == "ctx":
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, "b l t (h k) -> h b l t k", h=self.n_head)
            loc_attn = torch.einsum("hblk,hbltk->hblt", q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == "cond":
            spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
            spatial_weights = einops.rearrange(spatial_weights, "b l (h d) -> h b l d",
                                               h=self.spatial_n_head, d=self.spatial_dim + 1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, "1 b l d -> h b l d", h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum("hbld,bltd->hblt", spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, "b t -> h b l t", h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ["mul", "cond"]:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == "add":
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ["mul", "cond"]:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)

        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum("hblt,hbtv->hblv", fused_attn, v)
        output = einops.rearrange(output, "h b l v -> b l (h v)")
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn


class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        spatial_multihead=True,
        spatial_dim=5,
        spatial_attn_fusion="mul",
        # DHCT 閫忎紶
        use_dhct: bool = True, dhct_dim: int = 128, dhct_strength: float = 0.4,
        dhct_temp: float = 1.2, dhct_stop_grad_inputs: bool = False,
        dhct_use_log: bool = False, dhct_center_rows: bool = False, dhct_gate_bias: float = -6.0,
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            use_dhct=use_dhct, dhct_dim=dhct_dim, dhct_strength=dhct_strength,
            dhct_temp=dhct_temp, dhct_stop_grad_inputs=dhct_stop_grad_inputs,
            dhct_use_log=dhct_use_log, dhct_center_rows=dhct_center_rows, dhct_gate_bias=dhct_gate_bias,
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead, spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask, txt_embeds=memory[:, 0],
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        fused_attn_mask = self._build_dhct_bias(
            q=tgt2, k=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            base_attn_mask=memory_mask,
        )
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory, value=memory,
            attn_mask=fused_attn_mask, key_padding_mask=None,
            need_weights=True, average_attn_weights=False
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


@dataclass
class CMTConfig:
    # 绌洪棿瑙ｇ爜
    spatial_dec: bool = True
    spatial_multihead: bool = True
    spatial_dim: int = 5
    spatial_dist_norm: bool = True
    spatial_attn_fusion: str = "cond"
    # 缁撴瀯灏哄
    num_layers: int = 4
    obj_loc_encoding: str = "same_all"
    pairwise_rel_type: str = "center"
    hidden_size: int = 768
    num_attention_heads: int = 12
    dim_loc: int = 6
    # ---- DHCT controls ----
    use_dhct: bool = True
    dhct_dim: int = 64
    dhct_strength: float = 0.07
    dhct_temp: float = 1.2
    dhct_stop_grad_inputs: bool = False
    dhct_last_layer_only: bool = True
    dhct_last_n_layers: int = 4
    dhct_weak_guidance: bool = True
    dhct_use_log: bool = False
    dhct_center_rows: bool = False
    dhct_gate_bias: float = -6.0


class CMT(nn.Module):
    def __init__(self, config: CMTConfig):
        super().__init__()
        self.config = config

        # 鑻ョ敤鎴锋妸 dhct_weak_guidance=False锛屽垯鍙鐩栨垚鈥滃己涓€浜涒€濈殑榛樿
        if not self.config.dhct_weak_guidance:
            if not hasattr(self.config, "dhct_use_log"): self.config.dhct_use_log = True
            if not hasattr(self.config, "dhct_center_rows"): self.config.dhct_center_rows = True

        if self.config.spatial_dec:
            decoder_class = TransformerSpatialDecoderLayer
            kwargs = dict(
                spatial_dim=config.spatial_dim,
                spatial_multihead=config.spatial_multihead,
                spatial_attn_fusion=config.spatial_attn_fusion,
                use_dhct=config.use_dhct, dhct_dim=config.dhct_dim, dhct_strength=config.dhct_strength,
                dhct_temp=config.dhct_temp, dhct_stop_grad_inputs=config.dhct_stop_grad_inputs,
                dhct_use_log=config.dhct_use_log, dhct_center_rows=config.dhct_center_rows,
                dhct_gate_bias=config.dhct_gate_bias,
            )
        else:
            decoder_class = TransformerDecoderLayer
            kwargs = dict(
                use_dhct=config.use_dhct, dhct_dim=config.dhct_dim, dhct_strength=config.dhct_strength,
                dhct_temp=config.dhct_temp, dhct_stop_grad_inputs=config.dhct_stop_grad_inputs,
                dhct_use_log=config.dhct_use_log, dhct_center_rows=config.dhct_center_rows,
                dhct_gate_bias=config.dhct_gate_bias,
            )

        decoder_layer = decoder_class(
            config.hidden_size, config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation="gelu", **kwargs,
        )
        self.layers = _get_clones(decoder_layer, config.num_layers)

        loc_layer = nn.Sequential(nn.Linear(config.dim_loc, config.hidden_size), nn.LayerNorm(config.hidden_size))
        if self.config.obj_loc_encoding in ["same_0", "same_all"]:
            num_loc_layers = 1
        elif self.config.obj_loc_encoding == "diff_all":
            num_loc_layers = config.num_layers
        else:
            raise ValueError(f"Unknown obj_loc_encoding: {self.config.obj_loc_encoding}")
        self.loc_layers = _get_clones(loc_layer, num_loc_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calc_pairwise_locs(self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type="center"):
        if pairwise_rel_type == "mlp":
            obj_locs = torch.cat([obj_centers, obj_whls], 2)
            pairwise_locs = torch.cat(
                [einops.repeat(obj_locs, "b l d -> b l x d", x=obj_locs.size(1)),
                 einops.repeat(obj_locs, "b l d -> b x l d", x=obj_locs.size(1))], dim=3)
            return pairwise_locs

        pairwise_locs = einops.repeat(obj_centers, "b l d -> b l 1 d") - einops.repeat(obj_centers, "b l d -> b 1 l d")
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps)
        if self.config.spatial_dist_norm:
            max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
            norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, "b -> b 1 1")
        else:
            norm_pairwise_dists = pairwise_dists

        if self.config.spatial_dim == 1:
            return norm_pairwise_dists.unsqueeze(3)

        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
        if pairwise_rel_type == "center":
            pairwise_locs = torch.stack(
                [norm_pairwise_dists,
                 pairwise_locs[..., 2] / pairwise_dists,
                 pairwise_dists_2d / pairwise_dists,
                 pairwise_locs[..., 1] / pairwise_dists_2d,
                 pairwise_locs[..., 0] / pairwise_dists_2d], dim=3)
        elif pairwise_rel_type == "vertical_bottom":
            bottom_centers = torch.clone(obj_centers)
            bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
            bottom_pairwise_locs = einops.repeat(bottom_centers, "b l d -> b l 1 d") - einops.repeat(bottom_centers, "b l d -> b 1 l d")
            bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs**2, 3) + eps)
            bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps)
            pairwise_locs = torch.stack(
                [norm_pairwise_dists,
                 bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
                 bottom_pairwise_dists_2d / bottom_pairwise_dists,
                 pairwise_locs[..., 1] / pairwise_dists_2d,
                 pairwise_locs[..., 0] / pairwise_dists_2d], dim=3)

        if self.config.spatial_dim == 4:
            pairwise_locs = pairwise_locs[..., 1:]
        return pairwise_locs

    def forward(
        self,
        txt_embeds, txt_masks,
        obj_embeds, obj_locs, obj_masks,
        output_attentions=False, output_hidden_states=False,
    ):
        if self.config.spatial_dec:
            pairwise_locs = self.calc_pairwise_locs(
                obj_locs[:, :, :3], obj_locs[:, :, 3:], pairwise_rel_type=self.config.pairwise_rel_type,
            )

        out_embeds = obj_embeds
        all_hidden_states = [out_embeds]
        all_self_attn_matrices, all_cross_attn_matrices = [], []

        L = len(self.layers)
        for i, layer in enumerate(self.layers):
            # 浣嶇疆缂栫爜
            if self.config.obj_loc_encoding == "diff_all":
                query_pos = self.loc_layers[i](obj_locs)
                out_embeds = out_embeds + query_pos
            else:
                query_pos = self.loc_layers[0](obj_locs)
                if self.config.obj_loc_encoding == "same_all":
                    out_embeds = out_embeds + query_pos
                else:
                    if i == 0:
                        out_embeds = out_embeds + query_pos

            # 鎺у埗鍚敤 DHCT 鐨勫眰锛氫紭鍏?dhct_last_n_layers
            if hasattr(self.config, "dhct_last_n_layers") and (self.config.dhct_last_n_layers is not None):
                enable_this_layer = self.config.use_dhct and (i >= L - max(1, self.config.dhct_last_n_layers))
            else:
                enable_this_layer = self.config.use_dhct and (self.config.dhct_last_layer_only and (i == L - 1) or (not self.config.dhct_last_layer_only))

            if hasattr(layer, "use_dhct"):
                layer.use_dhct = enable_this_layer
                layer.dhct_strength = self.config.dhct_strength if enable_this_layer else 0.0

            # 鍓嶅悜
            if self.config.spatial_dec:
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds, pairwise_locs,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )
            else:
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )

            all_hidden_states.append(out_embeds)
            all_self_attn_matrices.append(self_attn_matrices)
            all_cross_attn_matrices.append(cross_attn_matrices)

        outs = {"obj_embeds": out_embeds}
        if output_hidden_states:
            outs["all_hidden_states"] = all_hidden_states
        if output_attentions:
            outs["all_self_attns"] = all_self_attn_matrices
            outs["all_cross_attns"] = all_cross_attn_matrices
        return outs

