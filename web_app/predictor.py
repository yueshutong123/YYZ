import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train import DualBranchClassifier
MODEL_PATH = os.path.join(BASE_DIR, '0.77best_model.pth')
DATA_DIR = os.path.join(BASE_DIR, 'dvlog-dataset')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEGMENT_LEN = 128
NUM_SEGMENTS = 8


def _load_model():
    model = DualBranchClassifier(visual_dim=128, acoustic_dim=96, num_classes=2, dropout_rate=0.55)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, ckpt


def _compute_norm_stats_from_data():
    import pandas as pd
    csv_path = os.path.join(DATA_DIR, 'labels.csv')
    df = pd.read_csv(csv_path)
    visual_all, acoustic_all = [], []
    for idx in df['index']:
        v = np.load(os.path.join(DATA_DIR, str(idx), f"{idx}_visual.npy"))
        a = np.load(os.path.join(DATA_DIR, str(idx), f"{idx}_acoustic.npy"))
        visual_all.append(v)
        acoustic_all.append(a)
    v_cat = np.concatenate(visual_all, axis=0)
    a_cat = np.concatenate(acoustic_all, axis=0)
    return {
        'vis_mean': v_cat.mean(axis=0, keepdims=True).astype(np.float32),
        'vis_std': v_cat.std(axis=0, keepdims=True).astype(np.float32) + 0.001,
        'ac_mean': a_cat.mean(axis=0, keepdims=True).astype(np.float32),
        'ac_std': a_cat.std(axis=0, keepdims=True).astype(np.float32) + 0.001,
    }


_model = None
_norm_stats = None
_ckpt_info = None


def get_model():
    global _model, _norm_stats, _ckpt_info
    if _model is None:
        _model, _ckpt_info = _load_model()
    if _norm_stats is None:
        _norm_stats = _compute_norm_stats_from_data()
    return _model, _norm_stats, _ckpt_info


def _extract_segments(arr, num_segs=NUM_SEGMENTS, seg_len=SEGMENT_LEN):
    src_len = arr.shape[0]
    dim = arr.shape[1]
    segs = np.zeros((num_segs, seg_len, dim), dtype=np.float32)
    if src_len > seg_len:
        max_start = src_len - seg_len
        starts = np.linspace(0, max_start, num_segs).astype(int)
        for i, s in enumerate(starts):
            segs[i] = arr[s:s + seg_len]
    elif src_len == seg_len:
        segs[:] = arr[None, :, :]
    else:
        for i in range(num_segs):
            segs[i, :src_len] = arr
    return segs


def predict(visual_features, acoustic_features):
    model, norm_stats, ckpt_info = get_model()

    visual = visual_features.astype(np.float32)
    acoustic = acoustic_features.astype(np.float32)

    visual = (visual - norm_stats['vis_mean']) / norm_stats['vis_std']
    acoustic = (acoustic - norm_stats['ac_mean']) / norm_stats['ac_std']

    all_probs = []
    num_rounds = 5
    with torch.no_grad():
        for _ in range(num_rounds):
            visual_segs = _extract_segments(visual)
            acoustic_segs = _extract_segments(acoustic)
            vs = torch.from_numpy(visual_segs).unsqueeze(0).to(DEVICE)
            acs = torch.from_numpy(acoustic_segs).unsqueeze(0).to(DEVICE)
            logits = model(vs, acs)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    avg_probs = np.mean(all_probs, axis=0)[0]

    depression_conf = float(avg_probs[1])
    normal_conf = float(avg_probs[0])
    predicted_class = 1 if depression_conf > normal_conf else 0

    return {
        'prediction': 'depression' if predicted_class == 1 else 'normal',
        'depression_confidence': round(depression_conf * 100, 2),
        'normal_confidence': round(normal_conf * 100, 2),
        'model_info': {
            'epoch': ckpt_info.get('epoch', 'unknown'),
            'val_accuracy': round(ckpt_info.get('val_acc', 0) * 100, 2),
            'val_f1': round(ckpt_info.get('val_f1', 0) * 100, 2),
        }
    }
