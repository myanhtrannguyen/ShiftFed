"""Sequential Federated Learning simulation for speedup comparison."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fl_common import (
    ALL_DOMAINS,
    ExperimentConfig,
    append_csv,
    build_model,
    evaluate_model,
    fedavg,
    make_loader,
    set_seed,
    state_dict_to_cpu,
    train_fixed_steps,
)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Sequential FL baseline over MNIST/SVHN/USPS.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", choices=["lenet5", "cnn", "mlp"], default="lenet5")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--log-dir", default="./outputs/fl_mpi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    return ExperimentConfig(**vars(parser.parse_args()))


def run_sequential(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    log_dir = Path(config.log_dir)
    log_path = log_dir / f"sequential_{config.model}_{config.rounds}r.csv"

    global_model = build_model(config.model).to(device)
    train_loaders = {domain: make_loader(config, domain, train=True) for domain in ALL_DOMAINS}
    test_loaders = {domain: make_loader(config, domain, train=False, shuffle=False) for domain in ALL_DOMAINS}

    rows = []
    total_start = time.perf_counter()
    for round_idx in range(1, config.rounds + 1):
        round_start = time.perf_counter()
        weighted_states = []
        client_metrics = []
        for client_idx, domain in enumerate(ALL_DOMAINS, start=1):
            local_model = build_model(config.model).to(device)
            local_model.load_state_dict(global_model.state_dict())
            metrics = train_fixed_steps(local_model, train_loaders[domain], config, device)
            weighted_states.append((state_dict_to_cpu(local_model.state_dict()), metrics["samples"]))
            client_metrics.append((client_idx, domain, metrics))

        global_model.load_state_dict(fedavg(weighted_states))
        eval_metrics = evaluate_model(global_model, test_loaders, device)
        round_time = time.perf_counter() - round_start

        base = {
            "mode": "sequential",
            "model": config.model,
            "round": round_idx,
            "round_time": round_time,
            "total_time": time.perf_counter() - total_start,
            **eval_metrics,
        }
        for client_idx, domain, metrics in client_metrics:
            rows.append(
                {
                    **base,
                    "rank": client_idx,
                    "domain": domain,
                    "client_loss": metrics["loss"],
                    "client_acc": metrics["acc"],
                    "compute_time": metrics["compute_time"],
                    "communication_time": 0.0,
                }
            )
        print(
            f"[SEQ] round={round_idx:03d} acc={eval_metrics['global_acc']:.4f} "
            f"loss={eval_metrics['global_loss']:.4f} time={round_time:.2f}s"
        )

    append_csv(log_path, rows)
    return log_path


if __name__ == "__main__":
    output = run_sequential(parse_args())
    print(f"Sequential log written to {output}")
