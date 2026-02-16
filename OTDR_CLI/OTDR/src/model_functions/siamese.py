from __future__ import annotations
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

@dataclass
class TrainConfig:
    # Model params (passed to base TCN)
    input_channels: int = 2
    num_classes: int = 8  # Not used for head, but for TCN sizing if needed
    levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.2
    
    # Training params
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    margin: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "outputs/siamese"
    
    # Feature params
    use_loss_reflectance: bool = False
    noise_level: float = 0.0

class SiameseNetwork(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward_once(self, x):
        # OTDR_TCN output: (metrics, positions)
        out, _ = self.base_model(x)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Label: 0 for similar, 1 for dissimilar
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SiameseDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, samples_per_epoch: int = 1000):
        self.X = X
        self.y = y # Class labels
        self.samples_per_epoch = samples_per_epoch
        
        # Pre-calculate indices for each class for fast retrieval
        self.class_indices = {}
        # Move to CPU for numpy indexing if it's on GPU
        y_np = y.cpu().numpy()
        unique_classes = np.unique(y_np)
        for cls in unique_classes:
            self.class_indices[cls] = np.where(y_np == cls)[0]

    def __getitem__(self, index):
        # We generate pairs on the fly, ignoring the 'index' mostly
        # to ensure randomness, or we can use index to select first sample.
        
        # Select first sample randomly
        idx1 = np.random.randint(0, len(self.X))
        sample1 = self.X[idx1]
        label1 = self.y[idx1].item()

        # 50% chance of same class, 50% chance of different class
        should_get_same_class = np.random.randint(0, 2)
        
        if should_get_same_class:
            # Positive pair (Same Class)
            # Choose another sample from same class
            indices = self.class_indices[label1]
            if len(indices) > 1:
                idx2 = np.random.choice(indices)
                # Ensure we don't pick the exact same index if possible (though for aug it might be fine)
            else:
                idx2 = indices[0]
            label = 0.0 # Similar
        else:
            # Negative pair (Different Class)
            possible_classes = list(self.class_indices.keys())
            if label1 in possible_classes:
                possible_classes.remove(label1)
            
            if not possible_classes:
                # If only one class exists, fallback to positive pair
                idx2 = np.random.choice(self.class_indices[label1])
                label = 0.0
            else:
                label2 = np.random.choice(possible_classes)
                idx2 = np.random.choice(self.class_indices[label2])
                label = 1.0 # Dissimilar

        sample2 = self.X[idx2]
        
        return sample1, sample2, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return self.samples_per_epoch

def train_siamese(
    processed_data_path: Path,
    config: TrainConfig,
    base_model_cls, # Class ref for TCN
    quiet: bool = False
):
    from data_helper import (
        load_raw_dataframe, 
        make_splits, 
        tensorise_splits, 
        fit_scaler, 
        measurement_columns,
        summarise_feature_layout
    )
    
    device = torch.device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {processed_data_path}")
    df = load_raw_dataframe(processed_data_path)
    train_df, val_df, test_df = make_splits(df)
    
    defaults = measurement_columns(train_df, include_loss_reflectance=config.use_loss_reflectance)
    layout = summarise_feature_layout(defaults)
    pos_count = int(layout["pos_count"])
    
    scaler = fit_scaler(train_df[defaults].values.astype(np.float32))
    
    tensors = tensorise_splits(
        train_df, 
        val_df, 
        test_df, 
        scaler, 
        measurement_override=defaults
    )
    
    if config.noise_level > 0.0:
        print(f"Injecting noise: {config.noise_level}")
        tensors["train"].X += torch.randn_like(tensors["train"].X) * config.noise_level
        tensors["val"].X += torch.randn_like(tensors["val"].X) * config.noise_level
    
    train_ds = SiameseDataset(tensors["train"].X, tensors["train"].y_class, samples_per_epoch=2000)
    val_ds = SiameseDataset(tensors["val"].X, tensors["val"].y_class, samples_per_epoch=500)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    # Helper to reshape inputs for OTDR_TCN
    def _reshape_input(xb: torch.Tensor) -> torch.Tensor:
        # Same logic as tcn._to_two_channel
        if xb.size(1) < pos_count + 1:
            raise ValueError(f"Expected at least {pos_count + 1} features")
        pos = xb[:, 1 : 1 + pos_count]
        extras = xb[:, 1 + pos_count :]
        seq_len = pos.size(1)
        channels = [pos]
        snr = xb[:, 0]
        channels.append(snr.unsqueeze(1).repeat(1, seq_len))
        if extras.numel() > 0:
            for idx in range(extras.size(1)):
                feat = extras[:, idx]
                channels.append(feat.unsqueeze(1).repeat(1, seq_len))
        return torch.stack(channels, dim=1)

    # 2. Model
    print("Building model...")
    n_classes = int(tensors["train"].y_class.max().item() + 1)
    
    # We need to broadcast the input channels correctly for OTDR_TCN
    # OTDR_TCN expects (B, C, L) where C is ~2 (Position + SNR)
    # But our tensors are (B, L, C) from tensorise_splits???
    # No, check tensorise_splits implementation again.
    # It calls _prepare_arrays which returns numpy arrays.
    # If it returns X as (N, F), then tensorise_splits makes it (N, F).
    # But OTDR_TCN expects (B, 2, L) and handles the conversion internally?
    # No, train_tcn calls `_to_two_channel` inside the loop.
    # siamese.py needs to do the same or the model needs to handle it.
    
    # Let's use the same logic as train_tcn:
    # We'll pass the raw input to SiameseNetwork, and SiameseNetwork...
    # Wait, SiameseNetwork just calls base_model.
    # So base_model (OTDR_TCN) expects (B, C, L).
    # But our dataset returns samples from X.
    # X is (N, F).
    # We need to reshape/transform X items to (C, L) before passing to model.
    # OR we can wrap OTDR_TCN to include the transform.
    
    # For now, let's assume we do the transform in the Dataset or Collate.
    # But SiameseDataset just returns X[i].
    
    # Let's look at train.py: train_tcn
    # It iterates dataloader, gets xb. 
    # xb = _to_two_channel(xb, pos_count=cfg.pos_count)
    # model(xb)
    
    # We must do this in Siamese training loop too!
    
    # First finish the instantiation fix:
    # Determine actual input channels after reshaping
    # It's 1 (Pos) + 1 (SNR) + num_extras
    in_ch = 2 + (len(defaults) - 1 - pos_count)
    base_model = base_model_cls(
        in_ch=in_ch,
        mid_ch=64,
        n_blocks=4,
        k=3,
        n_classes=n_classes,
        dropout=0.1
    ).to(device)
    
    siamese_net = SiameseNetwork(base_model).to(device)
    
    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=config.lr)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    start_time = time.perf_counter()
    
    print(f"Starting Siamese training on {device}...")
    
    for epoch in range(1, config.epochs + 1):
        siamese_net.train()
        train_loss = 0.0
        
        for batch_idx, (x1, x2, label) in enumerate(train_loader):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            
            # Reshape inputs
            x1 = _reshape_input(x1)
            x2 = _reshape_input(x2)
            
            optimizer.zero_grad()
            out1, out2 = siamese_net(x1, x2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        siamese_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, label in val_loader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                x1 = _reshape_input(x1)
                x2 = _reshape_input(x2)
                
                out1, out2 = siamese_net(x1, x2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the specialized encoder (the base model)
            torch.save(siamese_net.base_model.state_dict(), out_dir / "siamese_encoder.pt")
            
        if not quiet and epoch % 5 == 0:
            print(f"Epoch {epoch}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    return siamese_net
