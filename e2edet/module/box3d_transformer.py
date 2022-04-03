import math

import torch
from torch import nn

from .box_attention import Box3dAttention
from e2edet.utils.general import (
    flatten_with_shape,
    inverse_sigmoid,
    get_clones,
    get_activation_fn,
    get_proposal_pos_embed,
    normalize_period,
)


class Box3dTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        ref_size=4,
    ):
        super().__init__()

        encoder_layer = Box3dTransformerEncoderLayer(
            d_model, nhead, nlevel, dim_feedforward, dropout, activation
        )

        self.encoder = Box3dTransformerEncoder(
            d_model, encoder_layer, num_encoder_layers, num_queries
        )

        decoder_layer = Box3dTransformerDecoderLayer(
            d_model, nhead, nlevel, dim_feedforward, dropout, activation
        )

        self.decoder = Box3dTransformerDecoder(
            d_model, decoder_layer, num_decoder_layers
        )

        self.ref_size = ref_size

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, Box3dAttention):
                m._reset_parameters()

    def _create_ref_windows(self, tensor_list):
        angle_ratio = torch.FloatTensor(
            [
                0,
                2 * math.pi / 3,
                -2 * math.pi / 3,
                0,
                2 * math.pi / 3,
                -2 * math.pi / 3,
                0,
                2 * math.pi / 2,
            ]
        ).to(tensor_list[0])
        angle_ratio = normalize_period(angle_ratio, offset=0.5, period=math.pi * 2)
        h_ratio = torch.FloatTensor([self.ref_size] * 8).to(tensor_list[0])
        w_ratio = torch.FloatTensor([self.ref_size] * 8).to(tensor_list[0])

        ref_windows = []

        for tensor in tensor_list:
            b, _, h, w = tensor.shape

            y_embed = torch.arange(h, device=tensor.device, dtype=tensor.dtype) + 0.5
            y_embed = y_embed / h
            x_embed = torch.arange(w, device=tensor.device, dtype=tensor.dtype) + 0.5
            x_embed = x_embed / w

            y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")

            y_embed = y_embed.unsqueeze(0).expand(b, -1, -1)
            x_embed = x_embed.unsqueeze(0).expand(b, -1, -1)

            h_embed = torch.ones_like(y_embed).unsqueeze(-1) * h_ratio / h
            w_embed = torch.ones_like(x_embed).unsqueeze(-1) * w_ratio / w
            angle_embed = torch.ones_like(x_embed).unsqueeze(-1) * angle_ratio

            x_embed = x_embed.unsqueeze(-1).expand_as(angle_embed)
            y_embed = y_embed.unsqueeze(-1).expand_as(angle_embed)

            ref_box = torch.stack(
                [x_embed, y_embed, w_embed, h_embed, angle_embed], dim=-1
            ).flatten(1, 2)

            ref_windows.append(ref_box)

        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def forward(self, src, pos):
        assert pos is not None, "position encoding is required!"
        src_pos = []
        src_ref_windows = self._create_ref_windows(src)
        src, _, src_shape = flatten_with_shape(src, None)

        for pe in pos:
            b, c = pe.shape[:2]
            pe = pe.view(b, c, -1).transpose(1, 2)
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1)
        src_start_index = torch.cat(
            [src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]
        )

        output = self.encoder(src, src_pos, src_shape, src_start_index, src_ref_windows)
        out_embed, dec_embed, dec_ref_windows, dec_pos = output

        hs = self.decoder(
            dec_embed, dec_pos, out_embed, src_shape, src_start_index, dec_ref_windows
        )

        return hs, dec_ref_windows, out_embed, src_ref_windows


class Box3dTransformerEncoder(nn.Module):
    def __init__(self, d_model, encoder_layer, num_layers, num_queries):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

        self.detector = None
        self.num_queries = num_queries
        self.d_model = d_model
        self.enc_linear = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )

    def _get_enc_proposals(self, output, ref_windows):
        b, l = output.shape[:2]
        output_embed = output

        tmp_ref_windows = self.detector[0].bbox_embed(output_embed)
        num_references = self.detector[0].num_references

        tmp_ref_windows = tmp_ref_windows.view(b, l, num_references, 7)
        ref_windows = ref_windows[..., :num_references, :]

        tmp_ref_box, tmp_ref_height = tmp_ref_windows.split((5, 2), dim=-1)
        tmp_ref_box = tmp_ref_box + inverse_sigmoid(ref_windows)
        out_ref_windows = torch.cat([tmp_ref_box, tmp_ref_height], dim=-1).sigmoid()
        out_ref_windows = out_ref_windows.view(b, l * num_references, 7)

        ref_windows_valid = (
            (ref_windows[..., :2] > 0.001) & (ref_windows[..., :2] < 0.999)
        ).all(-1)
        src_mask = ~ref_windows_valid

        out_logits = (
            self.detector[0]
            .class_embed(output_embed)
            .view(b, l, num_references, -1)[..., 0]
        )
        out_logits = out_logits.masked_fill(src_mask, -65504.0)
        out_logits = out_logits.view(b, l * num_references)
        _, indexes = torch.topk(out_logits, self.num_queries, dim=1, sorted=False)

        indexes = indexes.unsqueeze(-1)
        out_ref_windows = torch.gather(
            out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1])
        )
        out_ref_windows = out_ref_windows.detach()

        pos = get_proposal_pos_embed(out_ref_windows[..., :2], self.d_model)
        size = get_proposal_pos_embed(out_ref_windows[..., 2:4], self.d_model)
        rad = get_proposal_pos_embed(out_ref_windows[..., [4, 4]], self.d_model)
        out_pos = pos + size + rad

        indexes = indexes.expand(-1, -1, output.shape[-1]).div(
            num_references, rounding_mode="floor"
        )
        out_embed = torch.gather(output_embed, 1, indexes)
        out_embed = self.enc_linear(out_embed.detach())

        return out_embed, out_ref_windows, out_pos

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows):
        output = src

        for layer in self.layers:
            output = layer(output, pos, src_shape, src_start_idx, ref_windows)

        out_embed, out_ref_windows, out_pos = self._get_enc_proposals(
            output, ref_windows
        )

        return output, out_embed, out_ref_windows, out_pos


class Box3dTransformerDecoder(nn.Module):
    def __init__(self, d_model, decoder_layer, num_layers):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)

    def forward(
        self, tgt, query_pos, memory, memory_shape, memory_start_idx, ref_windows
    ):
        output = tgt
        inter = []

        for layer in self.layers:
            output = layer(
                output, query_pos, memory, memory_shape, memory_start_idx, ref_windows
            )
            inter.append(output)

        return torch.stack(inter)


class Box3dTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = Box3dAttention(d_model, nlevel, nhead, with_rotation=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            src,
            src_shape,
            None,
            src_start_idx,
            None,
            ref_windows,
        )
        src = src + self.dropout1(src2[0])
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Box3dTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Box3dAttention(d_model, nlevel, nhead, with_rotation=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, query_pos, memory, memory_shape, memory_start_idx, ref_windows
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)

        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            self.with_pos_embed(tgt, query_pos),
            memory,
            memory_shape,
            None,
            memory_start_idx,
            None,
            ref_windows,
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
