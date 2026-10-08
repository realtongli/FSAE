"""Patch-Transformer reference observer: the official Medformer classifier (NeurIPS 2024), imported unchanged from
third_party/Medformer (commit 891f65b): models/Medformer.py::Model with layers.Embed.ListPatchEmbedding (cross-channel
patch tokens) and layers.SelfAttention_Family.MedformerLayer. Input x [B, seq_len, enc_in] (packet metadata: direction,
log IP length, log payload length, log inter-arrival time). The paper uses one patch length and no inter-granularity
attention, i.e. a plain patch Transformer.
"""
import os, sys, argparse
import torch, torch.nn as nn

_MED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'third_party', 'Medformer')
if _MED_DIR not in sys.path:
    sys.path.insert(0, _MED_DIR)
import importlib.util, types
if 'reformer_pytorch' not in sys.modules:  # SelfAttention_Family imports LSHSelfAttention (unused by Medformer); stub it
    _stub = types.ModuleType('reformer_pytorch'); _stub.LSHSelfAttention = object; sys.modules['reformer_pytorch'] = _stub
_spec = importlib.util.spec_from_file_location('medformer_official', os.path.join(_MED_DIR, 'models', 'Medformer.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)  # loads models/Medformer.py directly (models/__init__ pulls optional deps)
MedformerModel = _mod.Model  # official, unchanged


def medformer_configs(seq_len=64, enc_in=4, num_class=6, patch_len_list='8', d_model=128, d_ff=256, e_layers=6, n_heads=8,
                      dropout=0.1, augmentations='none', no_inter_attn=False, single_channel=False):
    return argparse.Namespace(task_name='classification', pred_len=0, output_attention=False, enc_in=enc_in,
                              single_channel=single_channel, patch_len_list=patch_len_list, seq_len=seq_len,
                              augmentations=augmentations, d_model=d_model, n_heads=n_heads, dropout=dropout,
                              no_inter_attn=no_inter_attn, d_ff=d_ff, activation='gelu', e_layers=e_layers, num_class=num_class)


class MetaCarrier(nn.Module):
    """Thin wrapper around the official model that returns the class logits."""

    def __init__(self, cfg):
        super().__init__()
        self.m = MedformerModel(cfg)

    def forward(self, x):
        m = self.m
        enc_out = m.enc_embedding(x)                       # list of [B, L_i, d]
        enc_out, _ = m.encoder(enc_out, attn_mask=None)
        out = m.dropout(m.act(enc_out)).reshape(enc_out.shape[0], -1)
        return m.projection(out)
