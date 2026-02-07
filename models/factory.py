"""
Model factory: build any model from a config dict.
"""

import torch.nn as nn

from .eegnet import EEGNet
from .deep_convnet import DeepConvNet
from .hybrid_cnn import HybridCNN


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate a model from config.

    Args:
        cfg: full config dict (with 'model' and 'data' keys).

    Returns:
        nn.Module
    """
    arch = cfg["model"]["architecture"]
    n_ch = cfg["data"]["n_channels"]
    n_t = cfg["data"]["n_timepoints"]
    n_cls = cfg["data"]["n_classes"]

    if arch == "eegnet":
        return EEGNet(
            n_channels=n_ch,
            n_timepoints=n_t,
            n_classes=n_cls,
            F1=cfg["model"].get("eegnet_F1", 8),
            F2=cfg["model"].get("eegnet_F2", 16),
            D=cfg["model"].get("eegnet_D", 2),
            kernel_length=cfg["model"].get("eegnet_kernel_length", 64),
            dropout_rate=cfg["model"].get("eegnet_dropout", 0.5),
        )
    elif arch == "deep_convnet":
        return DeepConvNet(
            n_channels=n_ch,
            n_timepoints=n_t,
            n_classes=n_cls,
            n_filters_time=cfg["model"].get("deep_n_filters_time", 25),
            n_filters_spat=cfg["model"].get("deep_n_filters_spat", 25),
            filter_time_length=cfg["model"].get("deep_filter_time_length", 10),
            dropout_rate=cfg["model"].get("hybrid_dropout", 0.5),
        )
    elif arch == "hybrid_cnn":
        return HybridCNN(
            n_channels=n_ch,
            n_timepoints=n_t,
            n_classes=n_cls,
            attention_type=cfg["model"].get("attention_type", "none"),
            temporal_filters=cfg["model"].get("hybrid_temporal_filters", 16),
            temporal_kernel=cfg["model"].get("hybrid_temporal_kernel", 64),
            spatial_filters=cfg["model"].get("hybrid_spatial_filters", 32),
            pool_size=cfg["model"].get("hybrid_pool_size", 8),
            dropout=cfg["model"].get("hybrid_dropout", 0.5),
            se_reduction=cfg["model"].get("se_reduction", 8),
            mha_n_heads=cfg["model"].get("mha_n_heads", 4),
            mha_d_model=cfg["model"].get("mha_d_model", 32),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
