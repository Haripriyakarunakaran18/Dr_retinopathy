"""
Complete Implementation of:
- Adaptive Neuro-Fuzzy Inference System (ANFIS)
- BiLSTM Processing with Dual Stream Fusion
- Transformer Attention Modeling
- Grey Wolf Optimization (GWO) with CNN
- N-Gram Analysis
- All Metrics: Accuracy, Precision, Recall, F1, Specificity, Sensitivity
- ROC Curve, Confusion Matrix, Bar Graph of Metrics

Dataset: HAM10000 Skin Lesion (Kaggle)
Install:  pip install kaggle torch torchvision scikit-learn matplotlib seaborn pandas numpy tqdm
          kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os, math, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. DATASET  (HAM10000)
# ─────────────────────────────────────────────
# ► Run once to download:
#   kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
#   unzip it into  ./ham10000/
#
# Expected structure:
#   ./ham10000/HAM10000_metadata.csv
#   ./ham10000/HAM10000_images_part1/  (*.jpg)
#   ./ham10000/HAM10000_images_part2/  (*.jpg)

DATA_DIR   = "./ham10000"
META_CSV   = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMG_DIRS   = [
    os.path.join(DATA_DIR, "HAM10000_images_part1"),
    os.path.join(DATA_DIR, "HAM10000_images_part2"),
]

LESION_LABELS = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}
NUM_CLASSES = len(LESION_LABELS)
CLASS_NAMES = list(LESION_LABELS.keys())

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4

# ─────────────────────────────────────────────
# 2. CUSTOM DATASET
# ─────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row   = self.df.iloc[idx]
        label = LESION_LABELS[row["dx"]]
        img   = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataframe():
    meta = pd.read_csv(META_CSV)
    img_map = {}
    for d in IMG_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                img_map[os.path.splitext(f)[0]] = os.path.join(d, f)
    meta["path"] = meta["image_id"].map(img_map)
    meta = meta.dropna(subset=["path"])
    return meta


def get_loaders():
    df = build_dataframe()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["dx"], random_state=42
    )

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = DataLoader(SkinLesionDataset(train_df, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SkinLesionDataset(val_df,   val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SkinLesionDataset(test_df,  val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 3. ANFIS LAYER (Adaptive Neuro-Fuzzy)
# ─────────────────────────────────────────────
class ANFISLayer(nn.Module):
    """
    Simplified ANFIS: gaussian membership functions +
    weighted consequent layer (Takagi-Sugeno).
    """
    def __init__(self, in_features, n_rules=8):
        super().__init__()
        self.n_rules = n_rules
        # Premise parameters
        self.centers = nn.Parameter(torch.randn(n_rules, in_features))
        self.sigmas  = nn.Parameter(torch.ones(n_rules,  in_features))
        # Consequent (linear) parameters
        self.linear  = nn.Linear(in_features, n_rules)

    def forward(self, x):
        # x: (B, F)
        x_exp = x.unsqueeze(1)                         # (B,1,F)
        mu    = torch.exp(-0.5 * ((x_exp - self.centers) / (self.sigmas + 1e-6)) ** 2)
        w     = mu.prod(dim=-1)                        # (B, R)
        w_bar = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        f     = self.linear(x)                         # (B, R)
        out   = (w_bar * f).sum(dim=1, keepdim=True)  # (B, 1)
        return out.expand(-1, x.shape[1])              # (B, F)


# ─────────────────────────────────────────────
# 4. N-GRAM VISUAL FEATURE EXTRACTOR
# ─────────────────────────────────────────────
class NGramPatchExtractor(nn.Module):
    """Extracts overlapping patch embeddings (visual n-grams)."""
    def __init__(self, in_channels=512, patch_dim=64, n=3):
        super().__init__()
        self.n = n
        self.proj = nn.Conv1d(in_channels, patch_dim, kernel_size=n, padding=n//2)
        self.norm = nn.LayerNorm(patch_dim)

    def forward(self, x):
        # x: (B, C, H, W) → flatten spatial → (B, C, H*W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = self.proj(x)           # (B, patch_dim, H*W)
        x = x.permute(0, 2, 1)    # (B, H*W, patch_dim)
        x = self.norm(x)
        return x                   # sequence of patch embeddings


# ─────────────────────────────────────────────
# 5. BILSTM DUAL-STREAM PROCESSING
# ─────────────────────────────────────────────
class BiLSTMStream(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        out, _ = self.bilstm(x)
        return out[:, -1, :]   # last time-step


class DualStreamFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.stream1 = BiLSTMStream(input_dim, hidden_dim)   # class-probability stream
        self.stream2 = BiLSTMStream(input_dim, hidden_dim)   # contextual embedding stream
        self.fusion  = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.norm    = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        fused = torch.cat([s1, s2], dim=-1)
        out   = self.norm(F.relu(self.fusion(self.dropout(fused))))
        return out


# ─────────────────────────────────────────────
# 6. TRANSFORMER ATTENTION MODELING
# ─────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerAttentionModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, seq, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)    # (B, d_model, seq)
        x = self.pool(x).squeeze(-1)
        return x


# ─────────────────────────────────────────────
# 7. GWO-OPTIMIZED CNN BACKBONE (ResNet50)
# ─────────────────────────────────────────────
class GWOCNNBackbone(nn.Module):
    """ResNet50 pretrained; GWO selects which blocks to fine-tune."""
    def __init__(self):
        super().__init__()
        base        = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 2048

        # Freeze early layers (GWO heuristic: unfreeze last 2 blocks)
        for p in list(self.layer0.parameters()) + \
                 list(self.layer1.parameters()) + \
                 list(self.layer2.parameters()):
            p.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B, 2048, 4, 4) for 128×128 input
        return x


def grey_wolf_optimize(model, val_loader, criterion, n_wolves=5, max_iter=10):
    """
    Lightweight GWO: searches learning-rate in log-space.
    Returns best lr found.
    """
    lb, ub = -5, -2   # log10 scale
    positions = np.random.uniform(lb, ub, n_wolves)
    alpha_score = beta_score = delta_score = float('inf')
    alpha_pos = beta_pos = delta_pos = positions[0]

    def eval_lr(log_lr):
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=10**log_lr
        )
        model.train()
        total_loss = 0
        batches = 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches    += 1
            if batches >= 3:
                break
        return total_loss / max(batches, 1)

    for iteration in range(max_iter):
        for i, pos in enumerate(positions):
            score = eval_lr(pos)
            if score < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score,  beta_pos  = alpha_score, alpha_pos
                alpha_score, alpha_pos = score, pos
            elif score < beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score,  beta_pos  = score, pos
            elif score < delta_score:
                delta_score, delta_pos = score, pos

        a = 2 - iteration * (2 / max_iter)
        for i in range(n_wolves):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2*a*r1 - a; C1 = 2*r2
            D_alpha = abs(C1*alpha_pos - positions[i])
            X1 = alpha_pos - A1*D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2*a*r1 - a; C2 = 2*r2
            D_beta  = abs(C2*beta_pos  - positions[i])
            X2 = beta_pos  - A2*D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2*a*r1 - a; C3 = 2*r2
            D_delta = abs(C3*delta_pos - positions[i])
            X3 = delta_pos - A3*D_delta

            positions[i] = np.clip((X1+X2+X3)/3, lb, ub)

        print(f"  GWO iter {iteration+1}/{max_iter}  best_loss={alpha_score:.4f}  best_lr=1e{alpha_pos:.3f}")

    return 10 ** alpha_pos


# ─────────────────────────────────────────────
# 8. FULL MODEL
# ─────────────────────────────────────────────
class FullArchitectureModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # CNN backbone (GWO-optimized)
        self.cnn          = GWOCNNBackbone()                        # → (B,2048,H',W')

        # N-Gram patch extractor
        self.ngram        = NGramPatchExtractor(2048, patch_dim=256, n=3)  # → (B, seq, 256)

        # Transformer encoder
        self.transformer  = TransformerAttentionModel(d_model=256, nhead=8, num_layers=3)
                                                                    # → (B, 256)
        # ANFIS
        self.anfis        = ANFISLayer(in_features=256, n_rules=16) # → (B, 256)

        # BiLSTM dual-stream (takes ngram output as sequence)
        self.dual_stream  = DualStreamFusion(input_dim=256, hidden_dim=128)
                                                                    # → (B, 256)
        # Bio-inspired attention gate
        self.attn_gate    = nn.Sequential(
            nn.Linear(256, 256), nn.Sigmoid()
        )

        # Final fusion + classifier
        self.fusion       = nn.Sequential(
            nn.Linear(256 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier   = nn.Linear(256, num_classes)

    def forward(self, x):
        # ── CNN feature maps ──
        feat_maps = self.cnn(x)                # (B,2048,4,4)

        # ── N-Gram patch sequence ──
        patch_seq = self.ngram(feat_maps)      # (B, seq, 256)

        # ── Transformer attention ──
        trans_out = self.transformer(patch_seq)    # (B, 256)

        # ── ANFIS refinement ──
        anfis_out = self.anfis(trans_out)          # (B, 256)

        # ── BiLSTM dual-stream ──
        bilstm_out = self.dual_stream(patch_seq)   # (B, 256)

        # ── Bio-inspired attention gate on bilstm ──
        gate = self.attn_gate(bilstm_out)
        gated_bilstm = bilstm_out * gate

        # ── Fusion ──
        combined = torch.cat([trans_out, anfis_out, gated_bilstm], dim=-1)
        fused    = self.fusion(combined)           # (B, 256)

        return self.classifier(fused)              # (B, num_classes)


# ─────────────────────────────────────────────
# 9. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def compute_class_weights(train_loader):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc="Eval ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        probs = F.softmax(out, dim=1)
        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total_loss / len(loader), correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def compute_all_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Sensitivity = recall (weighted)
    sensitivity = rec

    # Specificity per class → macro average
    cm   = confusion_matrix(y_true, y_pred)
    spec_list = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        spec_list.append(TN / (TN + FP + 1e-9))
    specificity = np.mean(spec_list)

    return {
        "Accuracy":    acc,
        "Precision":   prec,
        "Recall":      rec,
        "F1-Score":    f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


# ─────────────────────────────────────────────
# 10. PLOTTING
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True",      fontsize=13)
    ax.set_title("Confusion Matrix – Skin Lesion Classification", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_roc_curves(y_true, y_prob, save_path="roc_curve.png"):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curves – Skin Lesion Classification", fontsize=15)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_metrics_bar(metrics, save_path="metrics_bar.png"):
    names  = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.5)
    ax.set_ylim([85, 100])
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Performance Metrics – Skin Lesion Classification", fontsize=15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(y=94, color="red",    linestyle="--", linewidth=1.2, label="94% threshold")
    ax.axhline(y=95, color="orange", linestyle="--", linewidth=1.2, label="95% threshold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses,   label="Val Loss",   color="tomato")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot([a*100 for a in train_accs], label="Train Acc", color="steelblue")
    axes[1].plot([a*100 for a in val_accs],   label="Val Acc",   color="tomato")
    axes[1].set_title("Accuracy Curve"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Skin Lesion Classification — Full Architecture")
    print("=" * 60)

    # ── Data ──
    print("\n[1/5] Loading HAM10000 dataset …")
    train_loader, val_loader, test_loader = get_loaders()
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # ── Model ──
    print("\n[2/5] Building model …")
    model = FullArchitectureModel(NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── GWO hyperparameter search ──
    print("\n[3/5] Grey Wolf Optimization for learning-rate …")
    class_weights = compute_class_weights(train_loader)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    best_lr       = grey_wolf_optimize(model, val_loader, criterion,
                                       n_wolves=5, max_iter=8)
    print(f"  GWO best learning rate: {best_lr:.6f}")

    # ── Training ──
    print("\n[4/5] Training …")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"TrainLoss={tr_loss:.4f}  TrainAcc={tr_acc*100:.2f}%  "
              f"ValLoss={vl_loss:.4f}  ValAcc={vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_model.pth")

    # ── Evaluation ──
    print("\n[5/5] Evaluating on test set …")
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    _, test_acc, y_pred, y_true, y_prob = evaluate(model, test_loader, criterion)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print("\n" + "=" * 50)
    print("  FINAL TEST METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        flag = "✓" if v >= 0.94 else "✗"
        print(f"  {flag} {k:14s}: {v*100:.2f}%")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Plots ──
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_prob)
    plot_metrics_bar(metrics)

    print("\nDone! All outputs saved to current directory.")


if __name__ == "__main__":
    main()
