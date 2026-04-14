import torch
import torch.optim as optim
from tqdm import tqdm
import os
import time
import pandas as pd
from data_utils import split_dataset
from generative_model import ZernikeToPSFGenerator, ZernikeInverseDataset   # 新增的

def train_generative():
    save_dirs = ['weights', 'logs', 'results']
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    train_dir = "../dataset/def-onf-if/imgData-rr-z48"
    num_modes = 35
    epochs = 80          # 生成模型通常需要更多 epoch
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Splitting dataset...")
    train_idx, val_idx, _ = split_dataset(train_dir)

    train_dataset = ZernikeInverseDataset(train_dir, train_idx, num_modes)
    val_dataset   = ZernikeInverseDataset(train_dir, val_idx, num_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ZernikeToPSFGenerator(num_modes=num_modes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              steps_per_epoch=len(train_loader),
                                              epochs=epochs, pct_start=0.15)

    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    print(">>> Starting generative model training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Gen Train]")
        for coeffs, real_if in pbar:
            coeffs, real_if = coeffs.to(device), real_if.to(device)

            optimizer.zero_grad()
            gen_if = model(coeffs)
            loss = criterion(gen_if, real_if)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for coeffs, real_if in val_loader:
                coeffs, real_if = coeffs.to(device), real_if.to(device)
                gen_if = model(coeffs)
                val_loss += criterion(gen_if, real_if).item()

        avg_val = val_loss / len(val_loader)

        history['epoch'].append(epoch+1)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        print(f"Epoch {epoch+1}: Train MSE={avg_train:.6f} | Val MSE={avg_val:.6f}")

        pd.DataFrame(history).to_csv("./logs/generative_log.csv", index=False)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "./weights/generative_best.pth")
            print("    ★ Saved best generative model")

    print(">>> Generative model training finished!")

if __name__ == "__main__":
    train_generative()
