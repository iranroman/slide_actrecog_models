#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import re
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo
from copy import deepcopy

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.omni = torch.hub.load("facebookresearch/omnivore:main", model=cfg.MODEL.ARCH)

    def forward(self, x):
        y = self.omni(x, input_type="video")
        return y
