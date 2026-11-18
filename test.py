import argparse
import json
from pathlib import Path

import torch

from base_checkpoint import BASE_CHECKPOINT, load_base_checkpoint
from preprocessdata import load_coarse_mesh
from models import SubdBaseNet, SubdDispNet
from stage2_checkpoint import STAGE2_QUANTIZED_CHECKPOINT, load_quantized_disp
from tools.writeOBJ import writeOBJ


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
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    depth = int(params["numSubd"])
    base = SubdBaseNet(params)
    weights, normalization = load_base_checkpoint(
        folder / BASE_CHECKPOINT, model=base, require_normalization=True)
    meshes = load_coarse_mesh(args.mesh, depth, normalization)
    meshes.computeParameters(int(params["stage1_pe_dim"]))
    meshes.toDevice(device)

    base.load_state_dict(weights)
    base.to(device).eval()

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
        writeOBJ(str(path), vertices.cpu() * scale + center, mesh[level].F.cpu())

    print(f"Reconstruction results saved to {folder}")


if __name__ == "__main__":
    main()
