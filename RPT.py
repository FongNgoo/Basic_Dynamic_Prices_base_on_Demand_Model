#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_absolute_error
import warnings
from math import gcd
from functools import reduce
warnings.filterwarnings("ignore")


# # Config Set up

# In[2]:


def lcm(a, b):
    return abs(a*b) // gcd(a, b) if a and b else 0

def get_valid_h(base_h=128, R=3, kernel_count=3, num_heads=8):
    # h phải chia hết cho (R * kernel_count) và num_heads
    divisor = lcm(R * kernel_count, num_heads)
    for h in range(((base_h - 1) // divisor + 1) * divisor, base_h + 1000, divisor):
        if h >= base_h:
            return h
    raise ValueError(f"Không tìm được h >= {base_h} chia hết cho {divisor}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T, R = 60, 3
d = 64                    # 19 (room+env) + 45 (news PCA) = 64
num_heads = 8
kernel_count= 3
num_layers = 3
M = 3                     # MDN mixtures
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
PATIENCE = 10
h = h = get_valid_h(128, R, kernel_count, num_heads)

INPUT_NPZ = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model/Output/preprocessed_data.npz"
SCALERS_PKL = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model/Output/scalers.pkl"
MODEL_PATH = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model/Output/rpt_demand_best.pth"

print(f"[AUTO] h = {h}")
print(f"   → h % (R*{kernel_count}) = {h % 9} = 0")
print(f"   → h % num_heads = {h % 8} = 0")
print(f"[AUTO] h = {h} → per_room: {h//R}, per_kernel: {h//9}, head_dim: {h//8}")


# # Load Data

# In[3]:


data = np.load(INPUT_NPZ, allow_pickle=True)
X = data['X'].astype(np.float32)          # (N, T, R, d=64)
y_demand = data['y_demand'].astype(np.float32)  # (N, 3) ← [single, double, vip]
dates = data['dates']

with open(SCALERS_PKL, 'rb') as f:
    scalers = pickle.load(f)
price_scaler = scalers['price']   # ← chỉ có price, news

print(f"[INFO] X: {X.shape}, y_demand: {y_demand.shape}")

# Split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y_demand[:split], y_demand[split:]


# In[4]:


class RPTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # (N, 3)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                     # (T, R, d)
        y_val = self.y[idx]                 # (3,)
        news = x[:, 0, -45:]                # (T, 45) ← PCA thực tế
        x_feat = x[:, :, :19]               # (T, R, 19)
        return x_feat, y_val, news

train_dataset = RPTDataset(X_train, y_train)
val_dataset = RPTDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# # RPT model

# ## Encoder model

# In[5]:


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 3 == 0, f"out_channels={out_channels} phải chia hết 3"
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels//3, k, padding=k//2)
            for k in [1, 3, 5]
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.conv], dim=1)


# In[6]:


class CrossAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_model)
    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(q + out)


# In[7]:


class RPTEncoder(nn.Module):
    def __init__(self, embed_dim_news=45):
        super().__init__()
        d_room = 19
        self.h = h  # 144
        assert self.h % (R * 3) == 0 and self.h % num_heads == 0

        self.conv_per_room = nn.ModuleList([
            TemporalConv(d_room, self.h // R) for _ in range(R)
        ])
        self.news_proj = nn.Linear(embed_dim_news, self.h)
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(self.h, num_heads, batch_first=True)
            for _ in range(3)
        ])
        self.cross_attn = CrossAttention(self.h, num_heads)
        self.bilinear = nn.Parameter(torch.randn(self.h, self.h))
        self.norm = nn.LayerNorm(self.h)
        self.fusion = nn.Parameter(torch.ones(2))

    def forward(self, x, news_emb):
        B, T, R_in, _ = x.shape
        assert R_in == R

        # 1. Per-room conv
        h_list = [self.conv_per_room[r](x[:, :, r, :].transpose(1, 2)).transpose(1, 2)
                  for r in range(R)]
        h0 = torch.cat(h_list, dim=2)  # (B,T,h)

        # 2. Self-attention
        h = h0
        for attn in self.self_attn:
            h = self.norm(h + attn(h, h, h)[0])

        # 3. Cross-attention
        news_proj = self.news_proj(news_emb)
        h = self.cross_attn(h, news_proj, news_proj)

        # 4. BILINEAR INTERACTION – ĐÃ SỬA HOÀN TOÀN
        # M = tanh(h_i^T @ W_b @ h_j) → (B, T, T)
        M = torch.einsum('bti,btj,ij->bij', h, h, self.bilinear)
        M = torch.tanh(M)
        # h_bilinear = M @ h → (B, T, h)
        h_bilinear = torch.einsum('bij,btj->bti', M, h)

        # 5. Fusion
        w1, w2 = self.fusion.softmax(0)
        return w1 * h0 + w2 * h_bilinear


# In[8]:


class MDNHead(nn.Module):
    def __init__(self, input_dim, M=3, output_dim=3):
        super().__init__()
        self.M, self.output_dim = M, output_dim
        self.pi_net = nn.Sequential(nn.Linear(input_dim, M), nn.Softmax(dim=-1))
        self.mu_net = nn.Linear(input_dim, M * output_dim)
        self.sigma_net = nn.Sequential(nn.Linear(input_dim, M * output_dim), nn.Softplus())

    def forward(self, x):
        B = x.size(0)
        pi = self.pi_net(x)
        mu = self.mu_net(x).view(B, self.M, self.output_dim)
        sigma = self.sigma_net(x).view(B, self.M, self.output_dim) + 1e-6
        return pi, mu, sigma


# In[9]:


def mdn_loss(pi, mu, sigma, y):
    y = y.unsqueeze(1)  # (B,1,3)
    gaussian = torch.exp(-0.5 * ((y - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    likelihood = (pi.unsqueeze(-1) * gaussian).sum(dim=1)  # (B,3)
    return -torch.log(likelihood + 1e-8).mean()


# In[10]:


class RPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RPTEncoder(embed_dim_news=45)
        self.mdn = MDNHead(h, M=3, output_dim=3)
    def forward(self, x, news_emb):
        return self.mdn(self.encoder(x, news_emb).mean(dim=1))


# # Train Phase

# In[11]:


def train_rpt(model, train_loader, val_loader, optimizer, epochs=100, patience=10):
    train_losses, val_maes = [], []
    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb, newsb in train_loader:
            xb, yb, newsb = xb.to(DEVICE), yb.to(DEVICE), newsb.to(DEVICE)
            pi, mu, sigma = model(xb, newsb)
            loss = mdn_loss(pi, mu, sigma, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb, newsb in val_loader:
              xb, yb, newsb = xb.to(DEVICE), yb.to(DEVICE), newsb.to(DEVICE)
              pi, mu, sigma = model(xb, newsb)
              pred = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B,3)
              preds.extend(pred.cpu().numpy())
              trues.extend(yb.cpu().numpy())
        mae = mean_absolute_error(trues, preds)
        val_maes.append(mae)

        print(f"Epoch {epoch+1:3d} | Loss: {train_losses[-1]:.6f} | Val MAE: {mae:.3f} phòng")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOP] Epoch {epoch+1}")
                break

    print(f"\n[FINAL] Best MAE: {best_mae:.3f} phòng/phòng loại")
    return train_losses, val_maes, best_mae


# # Predict demand

# In[12]:


def predict_demand(model, last_window, news_emb, device=DEVICE):
    """
    Dự báo demand cho 3 loại phòng: single, double, vip
    Input:
        model: RPTModel (đã train)
        last_window: np.array shape (T, R, d_room) → T=60, R=3, d_room=19
        news_emb: np.array shape (T, embed_dim_news=45)
    Output:
        dict: {'single': int, 'double': int, 'vip': int}
    """
    model.eval()
    with torch.no_grad():
        # Chuẩn bị input
        x = torch.from_numpy(last_window).float().unsqueeze(0).to(device)  # (1, T, R, d)
        news = torch.from_numpy(news_emb).float().unsqueeze(0).to(device)   # (1, T, 45)

        # Forward
        pi, mu, sigma = model(x, news)  # pi: (1,3), mu: (1,3,3)

        # Tính kỳ vọng demand: E[demand] = Σ (pi * mu)
        pred_demand = (pi.unsqueeze(-1) * mu).sum(dim=1).cpu().numpy()[0]  # (3,)

    return {
        'single': int(round(pred_demand[0])),
        'double': int(round(pred_demand[1])),
        'vip':    int(round(pred_demand[2]))
    }


# # Main

# In[13]:


def main():
    print("=== RPT: DEMAND FORECASTING & DYNAMIC PRICING ===")
    model = RPTModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Huấn luyện
    train_rpt(model, train_loader, val_loader, optimizer)

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))

    # Lấy dữ liệu cuối cùng
    last_x = X[-1]  # (T, R, d_total)
    last_feat = last_x[:, :, :19]        # (T, R, 19) – room features
    last_news = last_x[:, 0, -45:]       # (T, 45) – news embedding (lấy từ phòng 0)

    # Dự báo demand
    pred = predict_demand(model, last_feat, last_news)  # ← CHỈ DEMAND

    # In kết quả
    tomorrow = pd.to_datetime(dates[-1]) + pd.Timedelta(days=1)
    print(f"\n=== DỰ BÁO DEMAND NGÀY {tomorrow:%d/%m/%Y} ===")
    print(f"   • Single: {pred['single']} phòng")
    print(f"   • Double: {pred['double']} phòng")
    print(f"   • VIP:    {pred['vip']} phòng")
    print(f"   • Tổng:   {pred['single'] + pred['double'] + pred['vip']} phòng")

    return model, pred


# In[14]:


if __name__ == "__main__":
    model, pred = main()

