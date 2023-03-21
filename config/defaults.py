#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.ARCH = "slowfast"

# Model name
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.NUM_CLASSES = [400, ]

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slowonly"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5



def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C
