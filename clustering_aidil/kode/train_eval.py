import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import MultimodalDefectDataset, compute_sensor_stats, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from models import (
    ImageOnlyClassifier, 
    SignalOnlyClassifier, 
    MultimodalLateFusion, 
    MultimodalCrossAttention
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            sensors = batch['sensor'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, sensors)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    avg_loss = total_loss / len(all_targets) if criterion is not None else 0.0
    return avg_loss, macro_f1, np.array(all_targets), np.array(all_preds)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        sensors = batch['sensor'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, sensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, macro_f1

def run_sanity_check(device):
    print("\n--- SANITY CHECK: Overfitting 1 small batch (32 samples) ---")
    set_seed(42)
    train_npz = np.load('data_uts_dl/klasifikasi_train.npz')
    train_meta = pd.read_csv('data_uts_dl/klasifikasi_train_meta.csv')
    
    # Pick 32 diverse samples
    sample_indices = np.arange(32)
    sub_meta = train_meta.iloc[sample_indices].reset_index(drop=True)
    dataset = MultimodalDefectDataset(
        images=train_npz['gambar'][sample_indices],
        sensors=train_npz['sensor'][sample_indices],
        meta_df=sub_meta,
        is_train=True,
        sensor_dropout_prob=0.0
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MultimodalLateFusion(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 41):
        loss, f1 = train_epoch(model, dataloader, optimizer, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/40 - Loss: {loss:.4f} - Macro-F1: {f1:.4f}")
            
    print(f"Sanity check completed! Final Loss: {loss:.4f}, Final Macro-F1: {f1:.4f}")
    assert loss < 0.1, "Model failed to overfit 1 batch!"
    print("Sanity check PASSED: Model capacity is verified.")

def main():
    parser = argparse.ArgumentParser(description="Task 1: Multimodal Defect Classification")
    parser.add_argument('--model-type', type=str, default='late_fusion', 
                        choices=['late_fusion', 'cross_attn', 'image_only', 'sensor_only'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--sensor-dropout', type=float, default=0.15)
    parser.add_argument('--per-line-norm', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--generate-submission', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.sanity_check:
        run_sanity_check(device)
        return

    set_seed(args.seed)
    
    # 1. Load data
    print("Loading data...")
    train_npz = np.load('data_uts_dl/klasifikasi_train.npz')
    train_meta = pd.read_csv('data_uts_dl/klasifikasi_train_meta.csv')
    
    # 2. Stratified split (preserving class distribution)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_meta)), 
        test_size=0.2, 
        stratify=train_meta['kelas'], 
        random_state=args.seed
    )
    
    train_sub_meta = train_meta.iloc[train_idx].reset_index(drop=True)
    val_sub_meta = train_meta.iloc[val_idx].reset_index(drop=True)
    
    # Compute sensor stats strictly on training split
    sensor_stats = compute_sensor_stats(
        train_npz['sensor'][train_idx], 
        train_sub_meta, 
        per_line=args.per_line_norm
    )
    
    train_ds = MultimodalDefectDataset(
        images=train_npz['gambar'][train_idx],
        sensors=train_npz['sensor'][train_idx],
        meta_df=train_sub_meta,
        sensor_stats=sensor_stats,
        is_train=True,
        sensor_dropout_prob=args.sensor_dropout,
        per_line_norm=args.per_line_norm
    )
    
    val_ds = MultimodalDefectDataset(
        images=train_npz['gambar'][val_idx],
        sensors=train_npz['sensor'][val_idx],
        meta_df=val_sub_meta,
        sensor_stats=sensor_stats,
        is_train=False,
        sensor_dropout_prob=0.0,
        per_line_norm=args.per_line_norm
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. Initialize Model
    if args.model_type == 'late_fusion':
        model = MultimodalLateFusion(num_classes=7)
    elif args.model_type == 'cross_attn':
        model = MultimodalCrossAttention(num_classes=7)
    elif args.model_type == 'image_only':
        model = ImageOnlyClassifier(num_classes=7)
    elif args.model_type == 'sensor_only':
        model = SignalOnlyClassifier(num_classes=7)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    print(f"\n--- Training {args.model_type} (Seed {args.seed}, Epochs {args.epochs}) ---")
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    
    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = f'checkpoints/{args.model_type}_seed{args.seed}.pt'
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, targets, preds = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train F1: {tr_f1:.4f} | Val Macro-F1: {val_f1:.4f}")
              
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    print(f"\nBest Validation Macro-F1: {best_val_f1:.4f} (Saved to {best_model_path})")
    
    # Load best model for detailed report
    model.load_state_dict(torch.load(best_model_path))
    _, final_f1, targets, preds = evaluate(model, val_loader, device, criterion)
    
    print("\n=== Detailed Classification Report (Validation) ===")
    print(classification_report(targets, preds, target_names=CLASSES, digits=4))
    
    # Domain Shift Analysis: Performance per production line
    print("=== Domain Shift Analysis (Per Production Line) ===")
    val_lines = val_sub_meta['lini'].values
    for line in ['A', 'B']:
        line_mask = (val_lines == line)
        if line_mask.sum() > 0:
            line_f1 = f1_score(targets[line_mask], preds[line_mask], average='macro')
            print(f"Line {line} (N={line_mask.sum()}): Macro-F1 = {line_f1:.4f}")
            
    # Save Learning Curves plot
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve - {args.model_type}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val Macro-F1')
    plt.title(f'Macro-F1 Curve - {args.model_type}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'figures/curve_{args.model_type}_seed{args.seed}.png', dpi=300)
    plt.close()
    
    # 4. Generate Submission if requested
    if args.generate_submission:
        print("\nGenerating Test Predictions...")
        test_npz = np.load('data_uts_dl/klasifikasi_test.npz')
        test_meta = pd.read_csv('data_uts_dl/klasifikasi_test_meta.csv')
        
        test_ds = MultimodalDefectDataset(
            images=test_npz['gambar'],
            sensors=test_npz['sensor'],
            meta_df=test_meta,
            sensor_stats=sensor_stats,
            is_train=False,
            sensor_dropout_prob=0.0,
            per_line_norm=args.per_line_norm
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        test_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                sensors = batch['sensor'].to(device)
                outputs = model(images, sensors)
                preds = outputs.argmax(dim=-1).cpu().numpy()
                test_preds.extend(preds)
                
        sub_df = pd.DataFrame({
            'id_unit': test_meta['id_unit'],
            'kelas': [IDX_TO_CLASS[p] for p in test_preds]
        })
        
        sub_df.to_csv('prediksi_klasifikasi.csv', index=False)
        print("Successfully saved predictions to 'prediksi_klasifikasi.csv'!")
        print(sub_df.head(10))

if __name__ == '__main__':
    main()
