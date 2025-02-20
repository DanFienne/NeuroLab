#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.nn.functional as F
from torch import nn
import torch


class InputEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()


class AtomConditioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_atom_pair_channels = 16
        self.per_token_channels = 768
        self.per_atom_channels = 128

        # single conditioning
        self.embed_ref_pos = nn.Linear(self.per_atom_channels, self.per_atom_channels, bias=False)
        self.embed_ref_mask = nn.Linear(self.per_atom_channels, self.per_atom_channels, bias=False)
        self.embed_ref_element = nn.Linear(128, self.per_atom_channels, bias=False)
        self.embed_ref_charge = nn.Linear(self.per_atom_channels, self.per_atom_channels, bias=False)
        self.embed_ref_atom_name = nn.Linear(64 * 4, self.per_atom_channels, bias=False)

    def forward(self, batch):
        ref_pos = batch['ref_pos']
        ref_mask = batch['ref_mask']
        ref_element = batch['ref_element']
        ref_charge = batch['ref_charge']
        ref_atom_name = batch['ref_atom_name_chars']

        # compute per-atom single conditioning
        act = self.embed_ref_pos(ref_pos)
        act += self.embed_ref_mask(ref_mask[:, :, None])
        act += self.embed_ref_element(F.one_hot(ref_element, num_classes=self.per_atom_channels).float())
        act += self.embed_ref_charge(torch.arcsinh(ref_charge)[:, :, None])

        # characters encoded as ASCII code minus 32, so need 64 classes, range (32, 96)
        atom_name_chars_one_hot = F.one_hot(ref_atom_name, num_classes=64).float()
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(atom_name_chars_one_hot.reshape(num_token, num_dense, -1))
        act *= ref_mask[:, :, None]
        return act


class AtomCrossAttentionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_atom_pair_channels = 16
        self.per_token_channels = 768
        self.per_atom_channels = 128

        self.per_atom_conditioning = AtomConditioning()
        self.layer_norm_trunk_single_cond = nn.LayerNorm(self.per_atom_channels, elementwise_affine=False)
        self.embed_trunk_single_cond = nn.Linear(self.per_atom_channels, self.per_atom_channels, bias=False)
        self.atom_position_2_features = nn.Linear(self.per_atom_channels, self.per_atom_channels, bias=False)

        self.row = nn.Linear(self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.row_relu = nn.ReLU()
        self.col = nn.Linear(self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.col_relu = nn.ReLU()




    def forward(self, token_atom_act, trunk_single_cond, trunk_pair_cond, batch):
        token_atom_single_cond = self.per_atom_conditioning(batch)
        token_atom_mask = batch['atom_mask']

        # attention: query
        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(
                self.layer_norm_trunk_single_cond(trunk_single_cond))
            query_single_cond = trunk_single_cond
        else:
            query_single_cond = token_atom_single_cond
        query_mask = token_atom_mask

        if token_atom_act is None:
            query_act = query_single_cond
        else:
            query_act = token_atom_act
            query_act = self.atom_position_2_features(query_act)
            query_act *= query_mask[..., None]
            query_act += query_single_cond

        # attention: key, value
        key_single_cond = query_single_cond
        key_mask = query_mask








