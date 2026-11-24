import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def compute_sensor_stats(sensors_array, df_meta=None, per_line=False):
    """
    Computes mean and std across valid (non-NaN) sensor readings.
    If per_line is True, computes separate stats for Line A and Line B.
    sensors_array: (N, 3, 128) — cast to float64 before aggregation to avoid float16 overflow.
    """
    if not per_line or df_meta is None:
        stats = {}
        for ch in range(3):
            ch_data = sensors_array[:, ch, :]
            valid = ch_data[~np.isnan(ch_data)].astype(np.float64)
            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 1.0
            stats[ch] = (mean, max(std, 1e-6))
        return stats
    else:
        stats = {'A': {}, 'B': {}}
        for line in ['A', 'B']:
            mask_line = (df_meta['lini'] == line).values
            for ch in range(3):
                ch_data = sensors_array[mask_line, ch, :]
                valid = ch_data[~np.isnan(ch_data)].astype(np.float64)
                mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
                std = float(np.std(valid)) if len(valid) > 0 else 1.0
                stats[line][ch] = (mean, max(std, 1e-6))
        return stats


class MultimodalDefectDataset(Dataset):
    def __init__(self,
                 images,
                 sensors,
                 meta_df,
                 sensor_stats=None,
                 is_train=False,
                 transform_img=None,
                 sensor_dropout_prob=0.15,
                 per_line_norm=False):
        """
        images: uint8 array (N, 64, 64)
        sensors: float16/float32 array (N, 3, 128)
        meta_df: DataFrame with metadata (id_unit, lini, shift, id_mesin, kelas (optional))
        """
        self.images = images
        self.sensors = sensors.astype(np.float32)
        self.meta_df = meta_df.reset_index(drop=True)
        self.is_train = is_train
        self.transform_img = transform_img
        self.sensor_dropout_prob = sensor_dropout_prob
        self.per_line_norm = per_line_norm

        # Precompute sensor stats if not provided
        if sensor_stats is None:
            self.sensor_stats = compute_sensor_stats(self.sensors, self.meta_df, per_line=per_line_norm)
        else:
            self.sensor_stats = sensor_stats

        # Targets if available
        if 'kelas' in self.meta_df.columns:
            self.labels = [CLASS_TO_IDX[k] for k in self.meta_df['kelas']]
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Process Image: (64, 64) -> (1, 64, 64) float32 in [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

        if self.transform_img is not None:
            img = self.transform_img(img)

        # 2. Process Sensor: (3, 128)
        raw_sensor = self.sensors[idx].copy()  # (3, 128)
        line = self.meta_df.loc[idx, 'lini'] if 'lini' in self.meta_df.columns else 'A'

        # Missing indicator mask: 1 if observed, 0 if NaN
        is_nan = np.isnan(raw_sensor)
        sensor_mask = (~is_nan).astype(np.float32)  # (3, 128)

        # Normalization
        norm_sensor = np.zeros_like(raw_sensor)
        stats = self.sensor_stats[line] if self.per_line_norm else self.sensor_stats
        for ch in range(3):
            mean, std = stats[ch]
            norm_sensor[ch] = np.where(is_nan[ch], 0.0, (raw_sensor[ch] - mean) / std)

        # Modality dropout during training: randomly drop sensor to train robust image-fallback
        if self.is_train and np.random.rand() < self.sensor_dropout_prob:
            norm_sensor = np.zeros_like(norm_sensor)
            sensor_mask = np.zeros_like(sensor_mask)

        sensor_tensor = torch.tensor(norm_sensor, dtype=torch.float32)
        mask_tensor = torch.tensor(sensor_mask, dtype=torch.float32)

        item = {
            'image': img,
            'sensor': sensor_tensor,
            'sensor_mask': mask_tensor,
            'id_unit': self.meta_df.loc[idx, 'id_unit'],
            'lini': line,
        }

        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
