import torch


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device=None) -> torch.device:
    if device is None:
        return get_default_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def empty_cache(device=None) -> None:
    dev = resolve_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
