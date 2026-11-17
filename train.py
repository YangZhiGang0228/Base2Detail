
import argparse
import json
import pickle
from pathlib import Path

import torch

from base_checkpoint import BASE_CHECKPOINT, load_base_checkpoint, save_base_checkpoint
from mesh_errors import normal_error, point2mesh_error
from models import SubdBaseNet, SubdDispNet, compute_vertex_normals
from stage2_checkpoint import (
    STAGE2_QUANTIZED_CHECKPOINT,
    load_quantized_disp,
    save_quantized_disp,
)
from tools.writeOBJ import writeOBJ


BEST_P2M_OBJ = "best_p2m.obj"


def unique_mesh_edges(faces):
    """Return each undirected triangle edge once, as used by the Stage-2 loss."""
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]],
                       faces[:, [2, 0]]), dim=0)
    return torch.unique(torch.sort(edges, dim=1).values, dim=0)


def scalar_gradient_loss(prediction, target, edges):
    """Mean squared difference of predicted and target scalar edge gradients."""
    residual = prediction - target
    return (residual[edges[:, 0]] - residual[edges[:, 1]]).square().mean()


def stage2_training_loss(prediction, target, edges, gradient_weight):
    """Return displacement MSE, edge-gradient MSE, and their weighted sum."""
    di_loss = torch.nn.functional.mse_loss(prediction, target)
    grad_loss = scalar_gradient_loss(prediction, target, edges)
    return di_loss, grad_loss, di_loss + gradient_weight * grad_loss


def set_global_seed(seed):
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_meshes(path, params):
    with open(path, "rb") as stream:
        meshes = pickle.load(stream)
    meshes.computeParameters(int(params["stage1_pe_dim"]))
    meshes.toDevice(params["device"])
    return meshes


def train_stage1(params, folder, epochs):
    set_global_seed(params["seed"])
    device = torch.device(params["device"])
    train_meshes = load_meshes(params["train_pkl"], params)
    if train_meshes.nM != 1:
        raise ValueError("Stage 1 expects one training mesh per checkpoint.")
    normalization = train_meshes.transforms[0]
    train_targets = [[mesh.V.to(device) for mesh in hierarchy] for hierarchy in train_meshes.meshes]

    model = SubdBaseNet(params).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(params["stage1_eta_min"]))
    mse = torch.nn.MSELoss()
    best = float("inf")
    log_every = int(params["stage1_log_every"])
    eval_every = int(params.get("stage1_eval_every", params.get("stage1_valid_eval_every", 10)))

    for epoch in range(epochs):
        model.train()
        train_error = torch.zeros((), device=device)
        for mesh_index in range(train_meshes.nM):
            outputs, _ = model(
                train_meshes.getInputData(mesh_index), mesh_index,
                train_meshes.hfList, train_meshes.poolMats,
                train_meshes.dofs, train_meshes.groupedLocalData, return_features=False)
            loss = sum(
                mse(output, train_targets[mesh_index][level][:len(output)])
                for level, output in enumerate(outputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_error += loss.detach()
        scheduler.step()

        evaluate = epoch % eval_every == 0 or epoch == epochs - 1
        terminal_text = "skipped"
        if evaluate:
            model.eval()
            terminal_error = 0.0
            with torch.no_grad():
                for mesh_index, targets in enumerate(train_targets):
                    outputs, _ = model(
                        train_meshes.getInputData(mesh_index), mesh_index,
                        train_meshes.hfList, train_meshes.poolMats,
                        train_meshes.dofs, train_meshes.groupedLocalData, return_features=False)
                    terminal_error += float(mse(outputs[-1], targets[-1][:len(outputs[-1])]))
            terminal_error /= train_meshes.nM
            terminal_text = "%.6e" % terminal_error
            if terminal_error < best:
                best = terminal_error
                save_base_checkpoint(model, normalization, folder / BASE_CHECKPOINT)

        if evaluate or epoch % log_every == 0:
            print("[Stage 1] epoch %d train %.6e terminal MSE %s" % (
                epoch, train_error / train_meshes.nM,
                terminal_text))

    print("[Stage 1] best final-level MSE %.6e" % best)


def train_stage2(params, folder, epochs):
    device = torch.device(params["device"])
    seed = params["p2m_seed"]
    set_global_seed(seed)
    train_meshes = load_meshes(params["train_pkl"], params)
    mesh_index = int(params["stage2_mesh_index"])
    level = int(params["numSubd"])

    net_base = SubdBaseNet(params).to(device)
    weights, normalization = load_base_checkpoint(folder / BASE_CHECKPOINT)
    if normalization is not None and any(
        not torch.equal(normalization[key], torch.as_tensor(train_meshes.transforms[mesh_index][key]).cpu())
        for key in ("center", "scale")
    ):
        raise ValueError("The Stage 2 training cache uses different mesh normalization from Stage 1.")
    net_base.load_state_dict(weights)
    net_base.requires_grad_(False)
    net_base.eval()

    with torch.no_grad():
        base_outputs, _ = net_base(
            train_meshes.getInputData(mesh_index), mesh_index,
            train_meshes.hfList, train_meshes.poolMats,
            train_meshes.dofs, train_meshes.groupedLocalData)
        base = base_outputs[-1].detach()
        target = train_meshes.meshes[mesh_index][level].V.to(device)
        faces = train_meshes.meshes[mesh_index][level].F.to(device)
        reference = train_meshes.orig_meshes[mesh_index].V.to(device)
        reference_faces = train_meshes.orig_meshes[mesh_index].F.to(device)

    model = SubdDispNet(params).to(device)
    features = model.anchor_features(base).detach()
    base_normals = compute_vertex_normals(base, faces).detach()
    target_scalar = ((target - base) * base_normals).sum(dim=1, keepdim=True)
    model.output_scale = float(target_scalar.abs().max())
    normalized_target = target_scalar / model.output_scale
    edges = unique_mesh_edges(faces)
    grad_weight = float(params["stage2_gradient_weight"])
    if grad_weight < 0:
        raise ValueError("stage2_gradient_weight must be non-negative")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["stage2_lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(params["stage2_eta_min"]))
    checkpoint = folder / STAGE2_QUANTIZED_CHECKPOINT
    candidate = Path(str(checkpoint) + ".candidate")
    scale = train_meshes.transforms[mesh_index]["scale"].cpu()
    center = train_meshes.transforms[mesh_index]["center"].cpu()
    bits = int(params["stage2_quantization_bits"])
    p2m_samples = int(params["stage2_p2m_samples"])
    best_p2m = float("inf")

    for epoch in range(epochs):
        model.train()
        scalar = model.net_disp(features)
        di_loss, grad_loss, loss = stage2_training_loss(
            scalar, normalized_target, edges, grad_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        evaluate = (epoch == 0 or (epoch + 1) % int(params["stage2_p2m_eval_every"]) == 0
                    or epoch == epochs - 1)
        p2m = None
        if evaluate:
            save_quantized_disp(model, candidate, bits=bits)
            quantized = SubdDispNet(params).to(device)
            load_quantized_disp(quantized, candidate, device)
            with torch.no_grad():
                reconstructed = quantized(base, F=faces)
                value, _, _ = point2mesh_error(
                    reconstructed, faces, reference, reference_faces,
                    sample_seed=seed, num_samples=p2m_samples)
            p2m = float(value)
            if p2m < best_p2m:
                best_p2m = p2m
                candidate.replace(checkpoint)
                writeOBJ(str(folder / BEST_P2M_OBJ),
                         reconstructed.cpu() * scale + center, faces.cpu())
                print("[Stage 2] epoch %d new best P2M %.6e, saved %s" % (
                    epoch, p2m, BEST_P2M_OBJ))

        if evaluate or epoch % int(params["stage2_log_every"]) == 0:
            print("[Stage 2] epoch %d di MSE %.6e gradient MSE %.6e "
                  "total %.6e quantized P2M %s" % (
                      epoch, float(di_loss), float(grad_loss), float(loss),
                      "%.6e" % p2m if p2m is not None else "skipped"))

    if candidate.exists():
        candidate.unlink()
    model = SubdDispNet(params).to(device)
    load_quantized_disp(model, checkpoint, device)
    with torch.no_grad():
        reconstructed = model(base, F=faces)
        final_p2m, _, _ = point2mesh_error(
            reconstructed, faces, reference, reference_faces,
            sample_seed=seed, num_samples=p2m_samples)
        final_normal = normal_error(
            reconstructed, faces, reference, reference_faces,
            sample_seed=seed, num_samples=int(params["stage2_normal_samples"]))
    print("[Stage 2] best P2M %.6e | normal error %.4f deg | saved %s" % (
        float(final_p2m), float(final_normal), checkpoint.name))


def main():
    parser = argparse.ArgumentParser(description='Train both AL_Q1 stages or a single stage.')
    parser.add_argument("folder", help='Directory containing hyperparameters.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--stage1-only", action="store_true")
    modes.add_argument("--stage2-only", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    with open(folder / "hyperparameters.json", "r", encoding="utf-8") as stream:
        params = json.load(stream)
    params["device"] = (
        params["device"] if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    stage1_epochs = int(params["stage1_epochs"])
    stage2_epochs = int(params["stage2_epochs"])
    if args.stage2_only:
        train_stage2(params, folder, stage2_epochs)
    elif args.stage1_only:
        train_stage1(params, folder, stage1_epochs)
    else:
        train_stage1(params, folder, stage1_epochs)
        train_stage2(params, folder, stage2_epochs)


if __name__ == "__main__":
    main()
