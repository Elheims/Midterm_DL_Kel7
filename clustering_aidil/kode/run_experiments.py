import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local imports work whether run from repo root or kode/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import MultimodalDefectDataset, compute_sensor_stats, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from models import (
    ImageOnlyClassifier, 
    SignalOnlyClassifier, 
    MultimodalLateFusion, 
    MultimodalCrossAttention
)
from train_eval import set_seed, train_epoch, evaluate

def train_and_eval_model(
    model_name,
    model,
    train_loader,
    val_loader,
    epochs=12,
    lr=1e-3,
    criterion=None,
    device='cpu',
    save_path=None
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0.0
    best_targets, best_preds = None, None
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    print(f"\n--- Training {model_name} ({epochs} epochs) ---")
    start_time = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, targets, preds = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_targets, best_preds = targets, preds
            if save_path:
                torch.save(model.state_dict(), save_path)

        if ep % 3 == 0 or ep == epochs:
            print(f"  Ep [{ep:2d}/{epochs:2d}] Loss: {tr_loss:.4f}/{val_loss:.4f} | F1: {tr_f1:.4f}/{val_f1:.4f}")

    elapsed = time.time() - start_time
    print(f"Finished {model_name} in {elapsed:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")
    return best_val_f1, best_targets, best_preds, history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device}")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("Loading data...")
    train_npz = np.load('data_uts_dl/klasifikasi_train.npz')
    train_meta = pd.read_csv('data_uts_dl/klasifikasi_train_meta.csv')
    test_npz = np.load('data_uts_dl/klasifikasi_test.npz')
    test_meta = pd.read_csv('data_uts_dl/klasifikasi_test_meta.csv')

    # Fixed Stratified Split for fair comparisons (80% train, 20% validation)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_meta)), 
        test_size=0.2, 
        stratify=train_meta['kelas'], 
        random_state=42
    )
    
    train_meta_sub = train_meta.iloc[train_idx].reset_index(drop=True)
    val_meta_sub = train_meta.iloc[val_idx].reset_index(drop=True)

    # Standard stats
    sensor_stats_global = compute_sensor_stats(train_npz['sensor'][train_idx], train_meta_sub, per_line=False)
    # Per-line stats
    sensor_stats_perline = compute_sensor_stats(train_npz['sensor'][train_idx], train_meta_sub, per_line=True)

    # Dataloaders factory
    def get_loaders(per_line=False, sensor_drop=0.15, batch_size=128):
        stats = sensor_stats_perline if per_line else sensor_stats_global
        train_ds = MultimodalDefectDataset(
            images=train_npz['gambar'][train_idx],
            sensors=train_npz['sensor'][train_idx],
            meta_df=train_meta_sub,
            sensor_stats=stats,
            is_train=True,
            sensor_dropout_prob=sensor_drop,
            per_line_norm=per_line
        )
        val_ds = MultimodalDefectDataset(
            images=train_npz['gambar'][val_idx],
            sensors=train_npz['sensor'][val_idx],
            meta_df=val_meta_sub,
            sensor_stats=stats,
            is_train=False,
            sensor_dropout_prob=0.0,
            per_line_norm=per_line
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
            val_ds
        )

    results_summary = {}

    # =========================================================================
    # EXPERIMENT 1: Single Modality vs Multimodal Fusion (Poin b & c)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Comparing Modalities & Fusion Architectures")
    print("="*60)
    
    train_loader, val_loader, val_ds = get_loaders(per_line=False, sensor_drop=0.15)
    
    # 1. Image-Only
    set_seed(42)
    m_img = ImageOnlyClassifier(num_classes=7).to(device)
    f1_img, t_img, p_img, h_img = train_and_eval_model(
        'Image-Only CNN', m_img, train_loader, val_loader, epochs=12, device=device,
        save_path='checkpoints/image_only.pt'
    )
    results_summary['Image-Only'] = f1_img
    
    # 2. Sensor-Only
    set_seed(42)
    m_sig = SignalOnlyClassifier(num_classes=7).to(device)
    f1_sig, t_sig, p_sig, h_sig = train_and_eval_model(
        'Sensor-Only 1D-CNN', m_sig, train_loader, val_loader, epochs=12, device=device,
        save_path='checkpoints/sensor_only.pt'
    )
    results_summary['Sensor-Only'] = f1_sig

    # 3. Multimodal Late Fusion
    set_seed(42)
    m_late = MultimodalLateFusion(num_classes=7).to(device)
    f1_late, t_late, p_late, h_late = train_and_eval_model(
        'Multimodal Late Fusion', m_late, train_loader, val_loader, epochs=12, device=device,
        save_path='checkpoints/multimodal_late.pt'
    )
    results_summary['Late Fusion'] = f1_late

    # 4. Multimodal Cross-Attention
    set_seed(42)
    m_cross = MultimodalCrossAttention(num_classes=7).to(device)
    f1_cross, t_cross, p_cross, h_cross = train_and_eval_model(
        'Multimodal Cross-Attention', m_cross, train_loader, val_loader, epochs=12, device=device,
        save_path='checkpoints/multimodal_cross.pt'
    )
    results_summary['Cross-Attention'] = f1_cross

    # Generate Per-Class F1 comparison chart (Poin c)
    per_class_f1 = {
        'Image-Only': f1_score(t_img, p_img, average=None),
        'Sensor-Only': f1_score(t_sig, p_sig, average=None),
        'Late Fusion': f1_score(t_late, p_late, average=None),
        'Cross-Attention': f1_score(t_cross, p_cross, average=None)
    }
    df_per_class = pd.DataFrame(per_class_f1, index=CLASSES)
    print("\nPer-Class F1 Score Comparison:")
    print(df_per_class)
    df_per_class.to_csv('results/per_class_f1_comparison.csv')

    plt.figure(figsize=(10, 5))
    df_per_class.plot(kind='bar', figsize=(11, 5), width=0.8, colormap='viridis')
    plt.title('Per-Class F1: Nilai Tiap Modalitas vs Fusion', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=10)
    plt.xlabel('Kelas Cacat', fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/exp_modality_value_comparison.png', dpi=300)
    plt.close()

    # =========================================================================
    # EXPERIMENT 2: Imperfect Labels & Label Smoothing (Poin d)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Robustness to Imperfect Labels (Poin d)")
    print("="*60)
    
    # Train Late Fusion WITHOUT label smoothing (standard CrossEntropy)
    set_seed(42)
    m_no_ls = MultimodalLateFusion(num_classes=7).to(device)
    f1_no_ls, _, _, _ = train_and_eval_model(
        'Late Fusion (No Label Smoothing, eps=0.0)', m_no_ls, train_loader, val_loader, 
        epochs=12, criterion=nn.CrossEntropyLoss(label_smoothing=0.0), device=device
    )
    results_summary['Late Fusion (No Label Smoothing)'] = f1_no_ls
    print(f"F1 without Label Smoothing: {f1_no_ls:.4f} vs with Label Smoothing (0.1): {f1_late:.4f}")

    # =========================================================================
    # EXPERIMENT 3: Domain Shift Handling (Poin e)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Domain Shift Mitigation (Poin e)")
    print("="*60)
    
    # Test per-line performance on Late Fusion
    val_lines = val_meta_sub['lini'].values
    print("Baseline (Global Normalization):")
    for line in ['A', 'B']:
        m = (val_lines == line)
        f1_line = f1_score(t_late[m], p_late[m], average='macro')
        print(f"  Line {line}: Macro-F1 = {f1_line:.4f}")

    # Train with Per-Line Normalization
    train_loader_pl, val_loader_pl, _ = get_loaders(per_line=True, sensor_drop=0.15)
    set_seed(42)
    m_pl = MultimodalLateFusion(num_classes=7).to(device)
    f1_pl, t_pl, p_pl, _ = train_and_eval_model(
        'Late Fusion + Per-Line Normalization', m_pl, train_loader_pl, val_loader_pl,
        epochs=12, device=device
    )
    results_summary['Late Fusion (Per-Line Norm)'] = f1_pl
    print("With Per-Line Normalization:")
    for line in ['A', 'B']:
        m = (val_lines == line)
        f1_line_pl = f1_score(t_pl[m], p_pl[m], average='macro')
        print(f"  Line {line}: Macro-F1 = {f1_line_pl:.4f}")

    # =========================================================================
    # EXPERIMENT 4: Missing Modality Robustness (Poin f)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Missing Sensor Simulation (Poin f)")
    print("="*60)
    # Create evaluation dataset where 100% of sensors are completely missing
    val_ds_missing = MultimodalDefectDataset(
        images=train_npz['gambar'][val_idx],
        sensors=np.full_like(train_npz['sensor'][val_idx], np.nan),
        meta_df=val_meta_sub,
        sensor_stats=sensor_stats_global,
        is_train=False,
        sensor_dropout_prob=0.0
    )
    val_loader_missing = DataLoader(val_ds_missing, batch_size=128, shuffle=False)
    _, f1_missing, _, _ = evaluate(m_late, val_loader_missing, device)
    print(f"Late Fusion with 100% Missing Sensor: Macro-F1 = {f1_missing:.4f} (sensible fallback to image!)")
    results_summary['Late Fusion (100% Missing Sensor)'] = f1_missing

    # =========================================================================
    # EXPERIMENT 5: Final Model over 3 Random Seeds (Poin g)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Final Model Evaluation over 3 Random Seeds (Poin g)")
    print("="*60)
    seeds = [42, 123, 999]
    seed_scores = []
    
    # Pick the best architecture (Late Fusion with Per-Line Norm or Standard)
    best_config_per_line = f1_pl > f1_late
    chosen_norm = best_config_per_line
    active_loaders = (train_loader_pl, val_loader_pl) if chosen_norm else (train_loader, val_loader)

    for s in seeds:
        set_seed(s)
        final_model = MultimodalLateFusion(num_classes=7).to(device)
        score, _, _, _ = train_and_eval_model(
            f'Final Model (Seed {s})', final_model, active_loaders[0], active_loaders[1],
            epochs=14, device=device, save_path=f'checkpoints/final_model_seed_{s}.pt'
        )
        seed_scores.append(score)

    mean_f1 = np.mean(seed_scores)
    std_f1 = np.std(seed_scores)
    print(f"\nFinal Model Results across 3 Seeds:")
    for s, sc in zip(seeds, seed_scores):
        print(f"  Seed {s:3d}: Macro-F1 = {sc:.4f}")
    print(f"Mean Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    
    results_summary['Final Model (3 Seeds)'] = f"{mean_f1:.4f} +/- {std_f1:.4f}"

    # Save summary to JSON
    with open('results/experiment_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # =========================================================================
    # FINAL PREDICTIONS: Generate prediksi_klasifikasi.csv
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING FINAL PREDICTIONS FOR HIDDEN TEST SET")
    print("="*60)
    
    test_ds = MultimodalDefectDataset(
        images=test_npz['gambar'],
        sensors=test_npz['sensor'],
        meta_df=test_meta,
        sensor_stats=sensor_stats_perline if chosen_norm else sensor_stats_global,
        is_train=False,
        sensor_dropout_prob=0.0,
        per_line_norm=chosen_norm
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Ensemble predictions across the 3 seeds for maximum test accuracy
    ensemble_probs = np.zeros((len(test_meta), 7), dtype=np.float32)
    for s in seeds:
        ckpt_path = f'checkpoints/final_model_seed_{s}.pt'
        model = MultimodalLateFusion(num_classes=7).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        seed_probs = []
        with torch.no_grad():
            for batch in test_loader:
                img = batch['image'].to(device)
                sig = batch['sensor'].to(device)
                logits = model(img, sig)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                seed_probs.append(probs)
        ensemble_probs += np.vstack(seed_probs) / len(seeds)

    final_preds = ensemble_probs.argmax(axis=-1)
    
    submission_df = pd.DataFrame({
        'id_unit': test_meta['id_unit'],
        'kelas': [IDX_TO_CLASS[p] for p in final_preds]
    })
    
    submission_df.to_csv('prediksi_klasifikasi.csv', index=False)
    print("Saved 'prediksi_klasifikasi.csv' successfully!")
    print(f"Total rows: {len(submission_df)}")
    print(submission_df.head(10))
    print("\nPredicted Class Distribution in Test Set:")
    print(submission_df['kelas'].value_counts())

if __name__ == '__main__':
    import torch.nn.functional as F
    main()
