"""ET-BERT classifier built directly from the authors' UER-py code (third_party/ET-BERT/uer, unchanged; commit d594706).
Mirrors fine-tuning/run_classifier.py::Classifier (embedding word_pos_seg -> TransformerEncoder -> [CLS] pooling ->
Linear+tanh -> Linear) with the authors' flags: --embedding word_pos_seg --encoder transformer --mask fully_visible,
bert_base_config.json (768/3072/12 layers/12 heads, dropout 0.1), post-LN, dense FFN.
Only differences: (1) loss is computed outside the model (label smoothing option); (2) forward returns logits only.
Pretrained weights: weights/etbert_pretrained_model.bin (all encoder/embedding keys load; target.* MLM/NSP heads unused).
"""
import os, sys, argparse
import torch, torch.nn as nn

_ETBERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'third_party', 'ET-BERT')
if _ETBERT_DIR not in sys.path:
    sys.path.insert(0, _ETBERT_DIR)
from uer.layers.embeddings import WordPosSegEmbedding  # noqa: E402
from uer.encoders.transformer_encoder import TransformerEncoder  # noqa: E402

VOCAB_SIZE = 60005


def default_args(dropout=0.1):
    return argparse.Namespace(emb_size=768, feedforward_size=3072, hidden_size=768, hidden_act='gelu', heads_num=12,
                              layers_num=12, dropout=dropout, max_seq_length=512, remove_embedding_layernorm=False,
                              mask='fully_visible', parameter_sharing=False, factorized_embedding_parameterization=False,
                              layernorm_positioning='post', relative_position_embedding=False, remove_transformer_bias=False,
                              remove_attention_scale=False, feed_forward='dense', layernorm='normal', relative_attention_buckets_num=32)


class ETBERTClassifier(nn.Module):
    def __init__(self, labels_num=6, dropout=0.1):
        super().__init__()
        args = default_args(dropout)
        self.embedding = WordPosSegEmbedding(args, VOCAB_SIZE)
        self.encoder = TransformerEncoder(args)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, labels_num)
        self.hidden_size = args.hidden_size

    def load_pretrained(self, path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        msg = self.load_state_dict(sd, strict=False)
        return dict(missing=list(msg.missing_keys), unexpected=[k for k in msg.unexpected_keys if not k.startswith('target.')])

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = output[:, 0, :]  # pooling 'first' ([CLS])
        output = torch.tanh(self.output_layer_1(output))
        return self.output_layer_2(output)
