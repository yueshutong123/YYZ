import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42
NUM_SEGMENTS = 8
SEGMENT_LEN = 128
EVAL_ROUNDS = 5
BASE_DIR = r'd:\Tools\yiyuzheng\dvlog-dataset'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_norm_stats(df, data_dir):
    visual_all, acoustic_all = [], []
    for idx in df['index']:
        v = np.load(os.path.join(data_dir, str(idx), f"{idx}_visual.npy"))
        a = np.load(os.path.join(data_dir, str(idx), f"{idx}_acoustic.npy"))
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


class DVLogDataset(Dataset):
    def __init__(self, df, data_dir, seg_len, num_segs, train_mode, norm_stats):
        self.df = df
        self.data_dir = data_dir
        self.seg_len = seg_len
        self.num_segs = num_segs
        self.train_mode = train_mode
        self.norm_stats = norm_stats
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _load(self, index):
        if index not in self._cache:
            visual_path = os.path.join(self.data_dir, str(index), f"{index}_visual.npy")
            acoustic_path = os.path.join(self.data_dir, str(index), f"{index}_acoustic.npy")
            visual = np.load(visual_path).astype(np.float32)
            acoustic = np.load(acoustic_path).astype(np.float32)
            if self.norm_stats is not None:
                visual = (visual - self.norm_stats['vis_mean']) / self.norm_stats['vis_std']
                acoustic = (acoustic - self.norm_stats['ac_mean']) / self.norm_stats['ac_std']
            self._cache[index] = (visual, acoustic)
        return self._cache[index]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        index = row['index']
        visual, acoustic = self._load(index)
        visual_segs = self._extract_segments(visual, self.train_mode)
        acoustic_segs = self._extract_segments(acoustic, self.train_mode)
        if self.train_mode:
            visual_segs = self._augment(visual_segs)
            acoustic_segs = self._augment(acoustic_segs)
        visual_segs = torch.from_numpy(visual_segs)
        acoustic_segs = torch.from_numpy(acoustic_segs)
        label = torch.tensor(1 if row['label'] == 'depression' else 0, dtype=torch.long)
        return visual_segs, acoustic_segs, label

    def _extract_segments(self, arr, random_sample=False):
        src_len = arr.shape[0]
        dim = arr.shape[1]
        segs = np.zeros((self.num_segs, self.seg_len, dim), dtype=np.float32)
        if src_len > self.seg_len:
            max_start = src_len - self.seg_len
            if random_sample:
                starts = np.sort(np.random.choice(max_start + 1, min(self.num_segs, max_start + 1), replace=False))
                if len(starts) < self.num_segs:
                    starts = np.linspace(0, max_start, self.num_segs).astype(int)
            else:
                starts = np.linspace(0, max_start, self.num_segs).astype(int)
            for i, s in enumerate(starts):
                segs[i] = arr[s:s + self.seg_len]
        elif src_len == self.seg_len:
            segs[:] = arr[None, :, :]
        else:
            for i in range(self.num_segs):
                segs[i, :src_len] = arr
        return segs

    def _augment(self, segs, noise_std=0.01):
        segs = segs.copy()
        for i in range(segs.shape[0]):
            mask_len = max(1, np.random.randint(1, self.seg_len // 8))
            start = np.random.randint(0, self.seg_len - mask_len + 1)
            segs[i, start:start + mask_len] *= np.random.uniform(0.7, 1.3, size=(mask_len, segs.shape[2]))
        segs += np.random.randn(*segs.shape).astype(np.float32) * noise_std
        return segs


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, conv_channels, output_dim):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels[-1], output_dim)

    def forward(self, x):
        B, K, T, C = x.shape
        x = x.view(B * K, T, C).permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = x.view(B, K, -1)
        x = x.mean(dim=1)
        return x


class DualBranchClassifier(nn.Module):
    def __init__(self, visual_dim=128, acoustic_dim=96, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.visual_encoder = ConvEncoder(136, [64, 128, 256], visual_dim)
        self.acoustic_encoder = ConvEncoder(25, [32, 64, 128], acoustic_dim)
        fused_dim = visual_dim + acoustic_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, visual_segs, acoustic_segs):
        vf = self.visual_encoder(visual_segs)
        af = self.acoustic_encoder(acoustic_segs)
        fused = torch.cat([vf, af], dim=1)
        return self.classifier(fused)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= max(1, self.warmup_steps):
            lr_scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_scale = self.min_lr / self.base_lrs[0] + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (1 + math.cos(math.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = blr * lr_scale

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    loss_sum = 0.0
    all_preds, all_labels = [], []
    for vs, ac, lb in loader:
        vs, ac, lb = vs.to(device), ac.to(device), lb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(vs, ac), lb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * vs.size(0)
        all_preds.extend(torch.argmax(model(vs, ac), dim=1).cpu().numpy())
        all_labels.extend(lb.cpu().numpy())
    return loss_sum / len(loader.dataset), accuracy_score(all_labels, all_preds)


def evaluate(model, dataset, criterion, device, num_rounds=EVAL_ROUNDS):
    model.eval()
    all_probs = []
    all_labels = None
    with torch.no_grad():
        for _ in range(num_rounds):
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            probs_list, labels_list = [], []
            for vs, ac, lb in loader:
                vs, ac, lb = vs.to(device), ac.to(device), lb.to(device)
                logits = model(vs, ac)
                probs_list.append(F.softmax(logits, dim=-1))
                labels_list.append(lb)
            epoch_probs = torch.cat(probs_list, dim=0)
            epoch_labels = torch.cat(labels_list, dim=0)
            all_probs.append(epoch_probs)
            all_labels = epoch_labels
    avg_probs = torch.stack(all_probs).mean(dim=0)
    preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    loss_val = F.cross_entropy(torch.log(avg_probs + 1e-8), all_labels).item()
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, average='binary')
    prec = precision_score(labels_np, preds, average='binary', zero_division=0)
    rec = recall_score(labels_np, preds, average='binary', zero_division=0)
    return loss_val, acc, f1, prec, rec


def main():
    set_seed(SEED)

    csv_path = os.path.join(BASE_DIR, 'labels.csv')
    save_path = os.path.join(os.path.dirname(BASE_DIR), 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Config: {NUM_SEGMENTS} segments x {SEGMENT_LEN} frames, {EVAL_ROUNDS} eval rounds")

    df = pd.read_csv(csv_path)
    train_df = df[df['fold'] == 'train'].reset_index(drop=True)
    valid_df = df[df['fold'] == 'valid'].reset_index(drop=True)

    print("Computing feature normalization from training set...")
    norm_stats = compute_norm_stats(train_df, BASE_DIR)

    print(f"Train: {len(train_df)} (depression: {(train_df['label']=='depression').sum()}, normal: {(train_df['label']=='normal').sum()})")
    print(f"Valid: {len(valid_df)} (depression: {(valid_df['label']=='depression').sum()}, normal: {(valid_df['label']=='normal').sum()})")

    train_dataset = DVLogDataset(train_df, BASE_DIR, SEGMENT_LEN, NUM_SEGMENTS, train_mode=True, norm_stats=norm_stats)
    valid_dataset = DVLogDataset(valid_df, BASE_DIR, SEGMENT_LEN, NUM_SEGMENTS, train_mode=False, norm_stats=norm_stats)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    model = DualBranchClassifier(visual_dim=128, acoustic_dim=96, num_classes=2, dropout_rate=0.55)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.1)

    num_epochs = 120
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 4
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    print(f"Steps: {total_steps} total, {warmup_steps} warmup")

    best_acc, best_f1, best_epoch = 0.0, 0.0, 0
    patience, no_improve = 25, 0

    print("\n" + "=" * 80)
    print("Training started")
    print("=" * 80)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, valid_dataset, criterion, device)

        lr = scheduler.get_lr()
        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR: {lr[0]:.6f}")
        print(f"  Train  | Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Valid  | Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")
        print(f"         | Prec: {val_prec:.4f}  Rec: {val_rec:.4f}")

        if val_acc > best_acc:
            best_acc, best_f1, best_epoch = val_acc, val_f1, epoch + 1
            no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'val_f1': val_f1}, save_path)
            print(f"  >>> Best model saved (Acc={val_acc:.4f})")
        else:
            no_improve += 1
        print("-" * 80)

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nDone! Best: Epoch {best_epoch}, Acc={best_acc:.4f}, F1={best_f1:.4f}")


if __name__ == '__main__':
    main()
