import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # Input: (B, 1, 64, 64)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResBlock2D(16, 32, stride=2)   # -> (B, 32, 32, 32)
        self.stage2 = ResBlock2D(32, 64, stride=2)   # -> (B, 64, 16, 16)
        self.stage3 = ResBlock2D(64, 128, stride=2)  # -> (B, 128, 8, 8)
        self.stage4 = ResBlock2D(128, 128, stride=2) # -> (B, 128, 4, 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_spatial=False):
        out = self.init_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out) # (B, 128, 4, 4)
        if return_spatial:
            B, C, H, W = out.shape
            # Return spatial tokens: (B, 16, 128)
            return out.view(B, C, H * W).permute(0, 2, 1)
        pooled = self.pool(out).flatten(1)
        return self.proj(pooled)

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, k=5, p=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=k, stride=1, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class SignalEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        # Input: (B, 3, 128)
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ResBlock1D(16, 32, stride=2, k=5, p=2)   # -> (B, 32, 64)
        self.stage2 = ResBlock1D(32, 64, stride=2, k=5, p=2)   # -> (B, 64, 32)
        self.stage3 = ResBlock1D(64, 128, stride=2, k=3, p=1)  # -> (B, 128, 16)
        self.stage4 = ResBlock1D(128, 128, stride=2, k=3, p=1) # -> (B, 128, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_seq=False):
        out = self.init_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out) # (B, 128, 8)
        if return_seq:
            # (B, 128, 8) -> (B, 8, 128)
            return out.permute(0, 2, 1)
        pooled = self.pool(out).squeeze(-1)
        return self.proj(pooled)

class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_classes=7, embed_dim=128):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, image, sensor=None):
        feat = self.encoder(image)
        return self.head(feat)

class SignalOnlyClassifier(nn.Module):
    def __init__(self, num_classes=7, embed_dim=128):
        super().__init__()
        self.encoder = SignalEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, image=None, sensor=None):
        feat = self.encoder(sensor)
        return self.head(feat)

class MultimodalLateFusion(nn.Module):
    """
    Fusion Strategy 1: Late Fusion (Concatenation of Image and Signal embeddings)
    """
    def __init__(self, num_classes=7, embed_dim=128):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.sig_encoder = SignalEncoder(embed_dim=embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensor):
        img_feat = self.img_encoder(image)
        sig_feat = self.sig_encoder(sensor)
        fused = torch.cat([img_feat, sig_feat], dim=-1)
        return self.classifier(fused)

class MultimodalCrossAttention(nn.Module):
    """
    Fusion Strategy 2: Cross-Attention Fusion
    Spatial tokens from Image attend to temporal tokens from Signal.
    """
    def __init__(self, num_classes=7, d_model=128, nheads=4):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim=d_model)
        self.sig_encoder = SignalEncoder(embed_dim=d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensor):
        # Image spatial tokens: (B, 16, 128)
        img_tokens = self.img_encoder(image, return_spatial=True)
        # Signal time tokens: (B, 8, 128)
        sig_tokens = self.sig_encoder(sensor, return_seq=True)
        
        # Cross Attention: Image queries Signal
        attn_out, _ = self.cross_attn(query=img_tokens, key=sig_tokens, value=sig_tokens)
        img_enhanced = self.norm1(img_tokens + attn_out)
        
        img_pooled = img_enhanced.mean(dim=1) # (B, 128)
        sig_pooled = sig_tokens.mean(dim=1)    # (B, 128)
        
        fused = torch.cat([img_pooled, sig_pooled], dim=-1)
        return self.classifier(fused)
