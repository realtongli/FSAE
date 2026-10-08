"""Tokenise MIRAGE-GenAI biflows for ET-BERT, following the authors' recipe
(third_party/ET-BERT/data_process/dataset_generation.py: bigram_generation / get_feature_flow).

Authors' flow-level recipe: for the first 5 packets, take the hex string of the frame from hex offset 76 (= byte 38,
i.e. inside the TCP header), split it into 2-hex-char units (cut(obj,1) with the %4 remainder rule -> bytes), and emit
bigrams unit[i]+unit[i+1] (4 hex chars = 2 overlapping bytes), at most 128 tokens per packet; fine-tuning then
truncates to seq_length=128 tokens ([CLS] + 127 tokens).
Adaptation: MIRAGE JSON has no frame/header bytes, so bigrams are built from the L4 payload of the first
5 payload-bearing packets (same packets as the YaTC input), 128 tokens per packet, then truncated to 128.
Vocab: third_party/ET-BERT/models/encryptd_vocab.txt (60,005 entries, [PAD]=0 [SEP]=1 [CLS]=2 [UNK]=3 [MASK]=4).
Output: data/derived/etbert_tokens.npz (src int32 [N,128], seg int8 [N,128]) aligned with genai_biflows.npz rows,
plus an SNI-masked variant (sni bytes -> 0x00 before tokenisation).
"""
import argparse, os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VOCAB = os.path.join(ROOT, 'third_party', 'ET-BERT', 'models', 'encryptd_vocab.txt')
SEQ, PER_PKT = 128, 128


def cut(obj, sec):  # verbatim from dataset_generation.py
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception:
        remanent_count = 0
    if remanent_count != 0:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def bigram_generation(packet_datagram, packet_len=64):  # verbatim logic, returns list of tokens
    generated_datagram = cut(packet_datagram, 1)
    toks, token_count = [], 0
    for i in range(len(generated_datagram)):
        if i != len(generated_datagram) - 1:
            token_count += 1
            if token_count > packet_len:
                break
            toks.append(generated_datagram[i] + generated_datagram[i + 1])
        else:
            break
    return toks


def official_tokenizer():
    """The tokenizer ET-BERT's run_classifier.py uses (UER BertTokenizer: basic tokenisation + WordPiece over the
    encrypted-traffic vocabulary), so bigrams missing from the vocabulary are split into sub-word pieces as in the
    original pipeline instead of becoming [UNK]."""
    sys.path.insert(0, os.path.join(ROOT, 'third_party', 'ET-BERT'))
    from uer.utils.tokenizers import BertTokenizer
    args = argparse.Namespace(spm_model_path=None, vocab_path=VOCAB)
    return BertTokenizer(args)


def main():
    vocab = {w.rstrip('\n'): i for i, w in enumerate(open(VOCAB, encoding='utf-8'))}
    tok = official_tokenizer()
    d = np.load(os.path.join(ROOT, 'data', 'derived', 'genai_biflows.npz'))
    pkt, plen, sni = d['pkt_bytes'], d['pkt_len'], d['pkt_sni']
    N = pkt.shape[0]
    out = {}
    for variant in ['plain', 'masksni']:
        src = np.zeros((N, SEQ), np.int32); seg = np.zeros((N, SEQ), np.int8); unk = 0; tot = 0
        for i in range(N):
            toks = []
            for p in range(pkt.shape[1]):
                L = int(min(plen[i, p], pkt.shape[2]))
                if L == 0:
                    continue
                b = pkt[i, p, :L].copy()
                if variant == 'masksni' and sni[i, p, 0] >= 0:
                    b[sni[i, p, 0]:sni[i, p, 1]] = 0
                toks += bigram_generation(b.tobytes().hex(), PER_PKT)
            pieces = tok.tokenize(' '.join(toks))
            ids = tok.convert_tokens_to_ids(['[CLS]'] + pieces)
            unk += sum(1 for t in pieces if t == '[UNK]'); tot += len(pieces)
            ids = ids[:SEQ]
            src[i, :len(ids)] = ids; seg[i, :len(ids)] = 1
        out[variant] = (src, seg)
        print(f'{variant}: unk rate {unk / max(tot, 1):.4f}, mean len {(seg > 0).sum(1).mean():.1f}, full-length frac {((seg > 0).sum(1) == SEQ).mean():.3f}')
    np.savez_compressed(os.path.join(ROOT, 'data', 'derived', 'etbert_tokens.npz'),
                        src=out['plain'][0], seg=out['plain'][1], src_masksni=out['masksni'][0], seg_masksni=out['masksni'][1])


if __name__ == '__main__':
    main()
