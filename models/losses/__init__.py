# Copyright (c) 2023 42dot. All rights reserved.
from .single_cam_loss import SingleCamLoss
from .multi_cam_loss import MultiCamLoss
from .depth_synthesis_loss import DepthSynLoss

__all__ = ['SingleCamLoss', 'MultiCamLoss', 'DepthSynLoss']