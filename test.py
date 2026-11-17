import argparse
import json
from pathlib import Path

import torch

from base_checkpoint import BASE_CHECKPOINT, load_base_checkpoint
from include import TestMeshes, tgp
from models import SubdBaseNet, SubdDispNet
from stage2_checkpoint import STAGE2_QUANTIZED_CHECKPOINT, load_quantized_disp


def main():
    parser = argparse.ArgumentParser(description="Decode a coarse mesh using trained AL_Q1 checkpoints.")
    parser.add_argument("folder", type=Path, help="Directory containing hyperparameters and checkpoints.")
    parser.add_argument("mesh", type=Path, help="Coarse input OBJ.")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    folder = args.folder.resolve()
    with (folder / "hyperparameters.json").open(encoding="utf-8") as stream:
        params = json.load(stream)

    device = params["device"] if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    depth = int(params["numSubd"])
    weights, normalization = load_base_checkpoint(
        folder / BASE_CHECKPOINT, require_normalization=True)
    meshes = TestMeshes([str(args.mesh)], depth, transforms=[normalization])
    meshes.computeParameters(int(params["stage1_pe_dim"]))
    meshes.toDevice(device)

    base = SubdBaseNet(params).to(device)
    base.load_state_dict(weights)
    base.eval()

    disp = None
    if args.stage != "1":
        disp = SubdDispNet(params).to(device)
        load_quantized_disp(disp, folder / STAGE2_QUANTIZED_CHECKPOINT, device)
        disp.eval()

    mesh = meshes.meshes[0]
    with torch.no_grad():
        outputs, _ = base(
            meshes.getInputData(0), 0, meshes.hfList, meshes.poolMats,
            meshes.dofs, meshes.groupedLocalData)
        if disp is not None:
            outputs[-1] = disp(outputs[-1], F=mesh[depth].F)

    center = meshes.transforms[0]["center"].cpu()
    scale = meshes.transforms[0]["scale"].cpu()
    for level, vertices in enumerate(outputs):
        path = folder / f"{args.mesh.stem}_subd{level}.obj"
        tgp.writeOBJ(str(path), vertices.cpu() * scale + center, mesh[level].F.cpu())

    print(f"Reconstruction results saved to {folder}")


if __name__ == "__main__":
    main()
