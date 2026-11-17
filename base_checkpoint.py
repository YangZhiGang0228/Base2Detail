import torch


BASE_CHECKPOINT = "netparams_base.dat"


def save_base_checkpoint(model, normalization, path):
    torch.save({
        "state_dict": model.state_dict(),
        "mesh_normalization": {
            "center": torch.as_tensor(normalization["center"]).detach().cpu(),
            "scale": torch.as_tensor(normalization["scale"]).detach().cpu(),
        },
    }, path)


def load_base_checkpoint(path, *, require_normalization=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    normalization = checkpoint.get("mesh_normalization")
    if normalization is None:
        if require_normalization:
            raise ValueError(
                "This Stage 1 checkpoint has no mesh normalization. "
                "Use a checkpoint saved with the training normalization.")
        return checkpoint, None

    center = normalization["center"]
    scale = normalization["scale"]
    if (center.shape != (3,) or scale.ndim != 0 or not torch.isfinite(center).all()
            or not torch.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid mesh normalization in the Stage 1 checkpoint.")
    return checkpoint["state_dict"], normalization
