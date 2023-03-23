from slide.models import build_model
from slide.datasets import loader

def slide(cfg):
    """
    Slide model over videos.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "slide")
