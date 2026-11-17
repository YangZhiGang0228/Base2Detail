
import argparse
import json
import os


NETWORK_SIZE_PROFILES = {
    "45": {"h_FNet": [14, 14], "h_matrixNet": [14, 14], "h_disp": [60, 60, 28]},
    "79": {"h_FNet": [24, 24], "h_matrixNet": [24, 24], "h_disp": [91, 91, 28]},
    "102": {"h_FNet": [32, 32], "h_matrixNet": [32, 32], "h_disp": [96, 96, 28]},
    "160": {"h_FNet": [48, 48], "h_matrixNet": [48, 48], "h_disp": [123, 123, 28]},
}


def main():
    parser = argparse.ArgumentParser(description="Write AL_Q1 training hyperparameters.")
    parser.add_argument("--train-pkl")
    parser.add_argument("--output", default="./jobs/net_cartoon_elephant/")
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()

    data = {
        "model_name": "bimba",
        "train_pkl": args.train_pkl or "./data_PKL/cartoon_elephant_train.pkl",
        "output_path": args.output,
        "device": "cuda",
        "seed": 2026,
        "p2m_seed": 12026,
        "Din": 6,
        "Dout": 32,
        "stage1_pe_dim": 16,
        "h_matrixNet": [32, 32],
        "h_FNet": [32, 32],
        "numSubd": args.levels,
        "stage1_epochs": 25000,
        "lr": 2e-3,
        "stage1_eta_min": 0.0,
        "stage1_eval_every": 10,
        "stage1_log_every": 100,
        "stage2_mesh_index": 0,
        "disp_n_anchor": 8,
        "disp_anchor_pe_dim": 8,
        "h_disp": [96, 96, 28],
        "stage2_epochs": 3000,
        "stage2_lr": 0.001,
        "stage2_eta_min": 0.0,
        "stage2_gradient_weight": 0.1,
        "stage2_quantization_bits": 10,
        "stage2_log_every": 100,
        "stage2_p2m_eval_every": 100,
        "stage2_p2m_samples": 1000000,
        "stage2_normal_samples": 1000000,
    }

    os.makedirs(data["output_path"], exist_ok=True)
    with open(os.path.join(data["output_path"], "hyperparameters.json"), "w",
              encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
