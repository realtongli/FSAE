"""Dataset access for the derived MIRAGE-GenAI-2025 arrays (see scripts/build_dataset.py)."""
import json, os
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLASS_NAMES = ['Chatgpt_text', 'Chatgpt_multi', 'Copilot_text', 'Copilot_multi', 'Gemini_text', 'Gemini_multi']
APP_NAMES = ['Chatgpt', 'Copilot', 'Gemini']
ACT_NAMES = ['text', 'multi']


class GenAIData:
    def __init__(self, root=ROOT, prefix='genai'):
        self.prefix = prefix
        self.n_act = {'genai': 2, 'ccma': 3}[prefix]
        self.n_app = {'genai': 3, 'ccma': 9}[prefix]
        d = np.load(os.path.join(root, 'data', 'derived', f'{prefix}_biflows.npz'))
        self.pkt_bytes, self.pkt_len, self.pkt_sni = d['pkt_bytes'], d['pkt_len'], d['pkt_sni']
        self.flat_bytes, self.flat_len, self.flat_sni = d['flat_bytes'], d['flat_len'], d['flat_sni']
        self.meta, self.meta_len = d['meta'], d['meta_len']
        self.y_app, self.y_act, self.y_joint = d['y_app'], d['y_act'], d['y_joint']
        self.session_id, self.device_id = d['session_id'], d['device_id']
        # compact joint labels (CCMA has app x activity cells that never occur, e.g. no 'chat' for GotoMeeting/Meet/Webex)
        uniq = np.unique(self.y_joint[self.y_joint >= 0]); self.joint_map = {int(u): i for i, u in enumerate(uniq)}
        self.y_joint = np.where(self.y_joint >= 0, np.searchsorted(uniq, np.maximum(self.y_joint, 0)), -1)
        self.n_joint = len(uniq)
        self.sessions = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_sessions.csv'))
        self.index = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_index.csv'))
        self.file2sid = dict(zip(self.sessions.file, self.sessions.session_id))

    def use_official_mfr(self, hdr=80, pay=240):
        """Rebuild the 5 x 320 rows as in YaTC's MFR: the first 5 packets of the flow, payload-free ones included, each row
        = hdr header bytes + pay payload bytes. MIRAGE releases no header bytes, so the header block is left at zero; the
        payload block holds the first `pay` bytes of that packet's L4 payload."""
        N, K, W = self.pkt_bytes.shape
        nb = np.zeros_like(self.pkt_bytes); ns = np.full_like(self.pkt_sni, -1); nl = np.zeros_like(self.pkt_len)
        for j in range(N):
            k = 0
            for r in range(min(K, int(self.meta_len[j]))):
                if self.meta[j, r, 2] > 0 and k < K:  # payload-bearing packet: take the next stored payload row
                    L = int(min(self.pkt_len[j, k], pay)); nb[j, r, hdr:hdr + L] = self.pkt_bytes[j, k, :L]; nl[j, r] = L
                    s, e = self.pkt_sni[j, k]
                    if s >= 0 and s < pay: ns[j, r] = (hdr + s, hdr + min(e, pay))
                    k += 1
        self.pkt_bytes, self.pkt_sni, self.pkt_len = nb, ns, nl; self.mfr_hdr = hdr

    def split_indices(self, split_json):
        sp = json.load(open(split_json))
        out = {}
        for k in ['train', 'val', 'test']:
            sids = np.array([self.file2sid[f] for f in sp[k]])
            out[k] = np.where(np.isin(self.session_id, sids) & (self.y_joint >= 0))[0]
        assert not (set(out['train']) & set(out['val'])) and not (set(out['train']) & set(out['test'])) and not (set(out['val']) & set(out['test']))
        return out, sp['meta']

    def controlled_indices(self):
        return np.where(self.y_joint < 0)[0]


class MFRDataset(Dataset):
    """YaTC-style 40x40 byte matrix built from the first 5 payload-bearing packets x 320 payload bytes
    (payload-only adaptation; original MFR used 80 header + 240 payload bytes per packet).
    Normalisation follows YaTC fine-tune.py: x/255 -> Normalize(mean=0.5, std=0.5)."""

    def __init__(self, data: GenAIData, idx, mask_sni=False, target='joint', mask_mode=None):
        self.data, self.idx, self.mask_sni = data, np.asarray(idx), mask_sni
        self.mask_mode = mask_mode or ('sni' if mask_sni else 'none')  # none | sni | clienthello (ECH: whole ClientHello record hidden)
        self.y = {'joint': data.y_joint, 'app': data.y_app, 'act': data.y_act}[target]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.data.pkt_bytes[j].astype(np.float32)  # (5, 320)
        if self.mask_mode == 'sni':
            for p in range(x.shape[0]):
                s, e = self.data.pkt_sni[j, p]
                if s >= 0:
                    x[p, s:e] = 0.0
        elif self.mask_mode == 'clienthello':
            o = getattr(self.data, 'mfr_hdr', 0)
            for p in range(x.shape[0]):
                if x[p, o] == 22 and x[p, o + 5] == 1:  # TLS handshake record carrying a ClientHello
                    x[p, o:] = 0.0
        x = (x.reshape(1, 40, 40) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(x), int(self.y[j])
