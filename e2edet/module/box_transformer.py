import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_attention import BoxAttention, InstanceAttention
from e2edet.utils.general import (
    flatten_with_shape,
    inverse_sigmoid,
    get_clones,
    get_activation_fn,
    get_proposal_pos_embed,
)


class BoxTransformer(nn.Module):
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
        use_mask=False,
        ref_size=4,
        residual_mode="v1",
    ):
        super().__init__()

        encoder_layer = BoxTransformerEncoderLayer(
            d_model, nhead, nlevel, dim_feedforward, dropout, activation
        )

        self.encoder = BoxTransformerEncoder(
            d_model, encoder_layer, num_encoder_layers, num_queries
        )

        decoder_layer = BoxTransformerDecoderLayer(
            d_model,
            nhead,
            nlevel,
            dim_feedforward,
            dropout,
            activation,
            use_mask,
            residual_mode,
        )

        self.decoder = BoxTransformerDecoder(
            decoder_layer, num_decoder_layers, use_mask=use_mask
        )

        self.ref_size = ref_size

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, (InstanceAttention, BoxAttention)):
                m._reset_parameters()

    def _create_ref_windows(self, tensor_list, mask_list):
        ref_windows = []

        eps = 1e-6
        for i, tensor in enumerate(tensor_list):
            if mask_list is not None:
                not_mask = ~(mask_list[i])
                y_embed = not_mask.cumsum(1, dtype=tensor.dtype)
                x_embed = not_mask.cumsum(2, dtype=tensor.dtype)

                size_h = not_mask[:, :, 0].sum(dim=-1, dtype=tensor.dtype)
                size_w = not_mask[:, 0, :].sum(dim=-1, dtype=tensor.dtype)
            else:
                size_h, size_w = tensor.shape[-2:]
                y_embed = torch.arange(
                    1, size_h + 1, dtype=tensor.dtype, device=tensor.device
                )
                x_embed = torch.arange(
                    1, size_w + 1, dtype=tensor.dtype, device=tensor.device
                )
                y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
                x_embed = x_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
                y_embed = y_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)

                size_h = torch.tensor(
                    [size_h] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )
                size_w = torch.tensor(
                    [size_w] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )

            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
            center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2)

            h_embed = self.ref_size / size_h
            w_embed = self.ref_size / size_w

            size = torch.stack([w_embed, h_embed], dim=-1)
            size = size.unsqueeze(1).expand_as(center)

            ref_box = torch.cat([center, size], dim=-1)
            ref_windows.append(ref_box)

        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _create_valid_ratios(self, src, masks):
        if masks is None:
            return None

        ratios = []
        for mask in masks:
            not_mask = ~mask
            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=src[0].dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=src[0].dtype)

            h, w = mask.shape[-2:]
            ratio_w = size_w / w
            ratio_h = size_h / h
            ratio = torch.stack([ratio_w, ratio_h], dim=-1)

            ratios.append(ratio)
        valid_ratios = (
            torch.stack(ratios, dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(-2)
        )

        return valid_ratios

    def forward(self, src, mask, pos):
        assert pos is not None, "position encoding is required!"
        if mask[0] is None:
            mask = None

        src_ref_windows = self._create_ref_windows(src, mask)
        src_valid_ratios = self._create_valid_ratios(src, mask)
        src, src_mask, src_shape = flatten_with_shape(src, mask)

        src_pos = []
        if pos[0] is not None:
            for pe in pos:
                b, c = pe.shape[:2]
                pe = pe.view(b, c, -1).transpose(1, 2)
                src_pos.append(pe)
            src_pos = torch.cat(src_pos, dim=1)
        else:
            assert self.adaptive_pe
            src_pos = None
        src_start_index = torch.cat(
            [src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]
        )

        output = self.encoder(
            src,
            src_pos,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            src_ref_windows,
        )
        out_embed, dec_embed, dec_ref_windows, dec_pos = output

        hs, roi = self.decoder(
            dec_embed,
            dec_pos,
            out_embed,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            dec_ref_windows,
        )

        return hs, roi, dec_ref_windows, out_embed, src_ref_windows, src_mask


class BoxTransformerEncoder(nn.Module):
    def __init__(self, d_model, encoder_layer, num_layers, num_queries):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

        self.detector = None
        self.num_queries = num_queries
        self.d_model = d_model
        self.enc_linear = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )

    def _get_enc_proposals(self, output, src_mask, ref_windows):
        output_embed = output

        ref_windows_valid = (
            (ref_windows[..., :2] > 0.01) & (ref_windows[..., :2] < 0.99)
        ).all(-1)
        if src_mask is not None:
            src_mask = src_mask & (~ref_windows_valid)
        else:
            src_mask = ~ref_windows_valid

        out_logits = self.detector[0].class_embed(output_embed)[..., 0]
        out_logits = out_logits.masked_fill(src_mask, -65504.0)
        _, indexes = torch.topk(out_logits, self.num_queries, dim=1, sorted=False)

        indexes = indexes.unsqueeze(-1)
        output_embed = torch.gather(
            output_embed, 1, indexes.expand(-1, -1, output.shape[-1])
        )
        out_embed = self.enc_linear(output_embed.detach())

        ref_windows = torch.gather(ref_windows, 1, indexes.expand(-1, -1, 4))
        tmp_ref_windows = self.detector[0].bbox_embed(output_embed)
        tmp_ref_windows += inverse_sigmoid(ref_windows)
        out_ref_windows = tmp_ref_windows.sigmoid().detach()

        pos = get_proposal_pos_embed(out_ref_windows[..., :2], self.d_model)
        size = get_proposal_pos_embed(out_ref_windows[..., 2:], self.d_model)
        out_pos = pos + size

        return out_embed, out_ref_windows, out_pos

    def forward(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_windows,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                pos,
                src_shape,
                src_mask,
                src_start_index,
                src_valid_ratios,
                ref_windows,
            )

        out_embed, out_ref_windows, out_pos = self._get_enc_proposals(
            output, src_mask, ref_windows
        )

        return output, out_embed, out_ref_windows, out_pos


class BoxTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, use_mask=False):
        super().__init__()

        self.detector = None
        self.layers = get_clones(decoder_layer, num_layers)
        self.use_mask = use_mask

    def forward(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_windows,
    ):
        output = tgt
        inter = []
        inter_roi = []

        for i, layer in enumerate(self.layers):
            # hack to return mask from the last layer
            if i == len(self.layers) - 1:
                layer.inferencing = False
                layer.multihead_attn.inferencing = False

            output, roi_feat = layer(
                output,
                query_pos,
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )
            inter.append(output)
            inter_roi.append(roi_feat)

        if self.inferencing:
            if self.use_mask:
                return inter[-1][None], inter_roi[-1][None]

            return inter[-1][None], None

        if self.use_mask:
            return torch.stack(inter), torch.stack(inter_roi)

        return torch.stack(inter), None


class BoxTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()

        self.self_attn = BoxAttention(d_model, nlevel, nhead)
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

    def forward(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_windows,
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            src,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_windows,
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class BoxTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        nlevel,
        dim_feedforward,
        dropout,
        activation,
        use_mask,
        residual_mode,
    ):
        super().__init__()
        assert residual_mode in ("v1", "v2")

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if use_mask:
            self.multihead_attn = InstanceAttention(d_model, nlevel, nhead, 14)
        else:
            self.multihead_attn = BoxAttention(d_model, nlevel, nhead)

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
        self.use_mask = use_mask
        self.residual_mode = residual_mode

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_windows,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_mask and not self.inferencing:
            tgt2, roi = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[:2]
        else:
            tgt2 = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[0]
            roi = None

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.use_mask and not self.inferencing:
            roi = tgt.unsqueeze(-2).unsqueeze(-2) + self.dropout2(roi)
            roi = self.norm2(roi)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.use_mask and not self.inferencing:
            if self.residual_mode == "v1":
                roi2 = self.linear2(self.dropout(self.activation(self.linear1(roi))))
                roi = roi + self.dropout3(roi2)
            elif self.residual_mode == "v2":
                roi = tgt.unsqueeze(-2).unsqueeze(-2) + self.dropout2(roi)
            roi = self.norm3(roi)

        return tgt, roi
