"""Test-only setup shared by both runners: no downloads, same random state.

Pretrained weights are replaced by timm's random initialisation (seeded), and
torch.hub is disabled, so the notebook and the package see identical models."""


def patch():
    import timm
    import torch

    torch.manual_seed(0)
    original = timm.create_model

    def create_model(*args, pretrained=False, **kwargs):
        return original(*args, pretrained=False, **kwargs)

    def no_hub(*args, **kwargs):
        raise RuntimeError("torch.hub was called during a test run")

    timm.create_model = create_model
    torch.hub.load = no_hub
