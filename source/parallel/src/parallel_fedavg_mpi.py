"""MPI Federated Averaging with domain-based client decomposition.

Rank 0 is the server. Ranks 1, 2, 3 own MNIST, SVHN, and USPS respectively.
The default path is synchronous FL with blocking bcast/gather. Use --async-mode
for a non-blocking Isend/Irecv server loop.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from mpi4py import MPI

from fl_common import (
    ALL_DOMAINS,
    DOMAIN_BY_RANK,
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


MODEL_TAG = 10
UPDATE_TAG = 20
STOP_TAG = 99


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Domain-decomposed MPI FedAvg.")
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
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--load-balance", action="store_true")
    parser.add_argument("--device", default="cpu")
    return ExperimentConfig(**vars(parser.parse_args()))


def ensure_world_size(comm: MPI.Comm) -> None:
    if comm.Get_size() < 2:
        if comm.Get_rank() == 0:
            print("This experiment expects at least 2 MPI processes: rank 0 server + at least 1 client.")
        raise SystemExit(2)


def run_sync_server(config: ExperimentConfig, comm: MPI.Comm, device: torch.device) -> Path:
    global_model = build_model(config.model).to(device)
    test_loaders = {domain: make_loader(config, domain, train=False, shuffle=False) for domain in ALL_DOMAINS}
    rows = []
    total_start = time.perf_counter()

    time_per_step = {}

    for round_idx in range(1, config.rounds + 1):
        round_start = time.perf_counter()
        state = state_dict_to_cpu(global_model.state_dict())
        
        assigned_steps = {r: config.local_steps for r in range(1, comm.Get_size())}
        if config.load_balance and len(time_per_step) > 0:
            fastest_step_time = min(time_per_step.values())
            target_time = fastest_step_time * config.local_steps
            for r in range(1, comm.Get_size()):
                if time_per_step.get(r, fastest_step_time) > 0:
                    assigned_steps[r] = max(10, int(target_time / time_per_step.get(r, fastest_step_time)))

        comm_start = time.perf_counter()
        comm.bcast({"round": round_idx, "state": state, "stop": False, "local_steps_dict": assigned_steps}, root=0)
        gathered = comm.gather(None, root=0)
        communication_time = time.perf_counter() - comm_start

        updates = [item for item in gathered[1:] if item is not None]
        for item in updates:
            r = item["rank"]
            if item["metrics"]["steps"] > 0:
                time_per_step[r] = item["metrics"]["compute_time"] / item["metrics"]["steps"]

        weighted_states = [(item["state"], item["metrics"]["samples"]) for item in updates]
        global_model.load_state_dict(fedavg(weighted_states))
        eval_metrics = evaluate_model(global_model, test_loaders, device)
        round_time = time.perf_counter() - round_start

        base = {
            "mode": "mpi_sync",
            "model": config.model,
            "round": round_idx,
            "round_time": round_time,
            "total_time": time.perf_counter() - total_start,
            "server_communication_time": communication_time,
            **eval_metrics,
        }
        for item in updates:
            rows.append(
                {
                    **base,
                    "rank": item["rank"],
                    "domain": item["domain"],
                    "client_loss": item["metrics"]["loss"],
                    "client_acc": item["metrics"]["acc"],
                    "compute_time": item["metrics"]["compute_time"],
                    "communication_time": item["communication_time"],
                }
            )
        print(
            f"[MPI-SYNC] round={round_idx:03d} acc={eval_metrics['global_acc']:.4f} "
            f"loss={eval_metrics['global_loss']:.4f} time={round_time:.2f}s"
        )

    comm.bcast({"round": config.rounds + 1, "state": state_dict_to_cpu(global_model.state_dict()), "stop": True}, root=0)
    log_path = Path(config.log_dir) / f"mpi_sync_{config.model}_{config.rounds}r.csv"
    append_csv(log_path, rows)
    return log_path


def run_sync_client(config: ExperimentConfig, comm: MPI.Comm, rank: int, device: torch.device) -> None:
    hostname = socket.gethostname()
    if "worker2" in hostname:
        domain = "svhn"
    elif "slave1" in hostname:
        domain = "mnist"
    else:
        domain = "usps"
        
    loader = make_loader(config, domain, train=True)
    local_model = build_model(config.model).to(device)
    while True:
        payload = comm.bcast(None, root=0)
        if payload["stop"]:
            break
        comm_start = time.perf_counter()
        local_model.load_state_dict(payload["state"])
        local_steps = payload.get("local_steps_dict", {}).get(rank, config.local_steps)
        metrics = train_fixed_steps(local_model, loader, config, device, local_steps=local_steps)
        update = {
            "rank": rank,
            "domain": domain,
            "round": payload["round"],
            "state": state_dict_to_cpu(local_model.state_dict()),
            "metrics": metrics,
            "communication_time": time.perf_counter() - comm_start - metrics["compute_time"],
        }
        comm.gather(update, root=0)


def run_async_server(config: ExperimentConfig, comm: MPI.Comm, device: torch.device) -> Path:
    global_model = build_model(config.model).to(device)
    test_loaders = {domain: make_loader(config, domain, train=False, shuffle=False) for domain in ALL_DOMAINS}
    rows = []
    total_start = time.perf_counter()

    send_requests = {}
    time_per_step = {}
    client_ranks = range(1, comm.Get_size())
    for rank in client_ranks:
        send_requests[rank] = comm.isend({"round": 1, "state": state_dict_to_cpu(global_model.state_dict()), "stop": False, "local_steps": config.local_steps}, dest=rank, tag=MODEL_TAG)

    completed_updates: List[Tuple[Dict[str, object], float]] = []
    target_updates = config.rounds * len(client_ranks)
    next_round_by_rank = {rank: 1 for rank in client_ranks}

    while len(completed_updates) < target_updates:
        status = MPI.Status()
        update = comm.recv(source=MPI.ANY_SOURCE, tag=UPDATE_TAG, status=status)
        recv_time = time.perf_counter()
        rank = int(update["rank"])
        completed_updates.append((update, recv_time))

        if update["metrics"]["steps"] > 0:
            time_per_step[rank] = update["metrics"]["compute_time"] / update["metrics"]["steps"]

        global_model.load_state_dict(fedavg([(state_dict_to_cpu(global_model.state_dict()), 1.0), (update["state"], update["metrics"]["samples"])]))

        eval_metrics = evaluate_model(global_model, test_loaders, device)
        rows.append(
            {
                "mode": "mpi_async",
                "model": config.model,
                "round": update["round"],
                "rank": rank,
                "domain": update["domain"],
                "client_loss": update["metrics"]["loss"],
                "client_acc": update["metrics"]["acc"],
                "compute_time": update["metrics"]["compute_time"],
                "communication_time": update["communication_time"],
                "round_time": recv_time - total_start,
                "total_time": recv_time - total_start,
                **eval_metrics,
            }
        )
        print(
            f"[MPI-ASYNC] update={len(completed_updates):03d}/{target_updates} "
            f"rank={rank} acc={eval_metrics['global_acc']:.4f}"
        )

        next_round_by_rank[rank] += 1
        
        # Đợi request gửi trước đó hoàn tất để tránh bị giải phóng vùng nhớ
        if rank in send_requests:
            send_requests[rank].Wait()

        if next_round_by_rank[rank] <= config.rounds:
            assigned_steps = config.local_steps
            if config.load_balance and len(time_per_step) > 0:
                fastest_step_time = min(time_per_step.values())
                if time_per_step.get(rank, fastest_step_time) > 0:
                    target_time = fastest_step_time * config.local_steps
                    assigned_steps = max(10, int(target_time / time_per_step.get(rank, fastest_step_time)))

            send_requests[rank] = comm.isend(
                {
                    "round": next_round_by_rank[rank],
                    "state": state_dict_to_cpu(global_model.state_dict()),
                    "stop": False,
                    "local_steps": assigned_steps,
                },
                dest=rank,
                tag=MODEL_TAG,
            )
        else:
            send_requests[rank] = comm.isend({"round": next_round_by_rank[rank], "state": None, "stop": True}, dest=rank, tag=STOP_TAG)
    
    # Đợi toàn bộ tin nhắn đã được gửi đi an toàn trước khi thoát server
    for req in send_requests.values():
        req.Wait()
        
    log_path = Path(config.log_dir) / f"mpi_async_{config.model}_{config.rounds}r.csv"
    append_csv(log_path, rows)
    return log_path


def run_async_client(config: ExperimentConfig, comm: MPI.Comm, rank: int, device: torch.device) -> None:
    hostname = socket.gethostname()
    if "worker2" in hostname:
        domain = "svhn"
    elif "slave1" in hostname:
        domain = "mnist"
    else:
        domain = "usps"
        
    loader = make_loader(config, domain, train=True)
    local_model = build_model(config.model).to(device)
    while True:
        status = MPI.Status()
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if payload["stop"] or status.Get_tag() == STOP_TAG:
            break
        comm_start = time.perf_counter()
        local_model.load_state_dict(payload["state"])
        local_steps = payload.get("local_steps", config.local_steps)
        metrics = train_fixed_steps(local_model, loader, config, device, local_steps=local_steps)
        update = {
            "rank": rank,
            "domain": domain,
            "round": payload["round"],
            "state": state_dict_to_cpu(local_model.state_dict()),
            "metrics": metrics,
            "communication_time": time.perf_counter() - comm_start - metrics["compute_time"],
        }
        request = comm.isend(update, dest=0, tag=UPDATE_TAG)
        request.Wait()


def main() -> None:
    config = parse_args()
    comm = MPI.COMM_WORLD
    ensure_world_size(comm)
    rank = comm.Get_rank()
    set_seed(config.seed + rank)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    hostname = socket.gethostname()
    
    if rank == 0:
        print(f"[INIT] Rank {rank} started on host: {hostname} (SERVER)", flush=True)
    else:
        if "worker2" in hostname:
            domain_print = "SVHN"
        elif "slave1" in hostname:
            domain_print = "MNIST"
        else:
            domain_print = "USPS"
        print(f"[INIT] Rank {rank} started on host: {hostname} (Dataset: {domain_print})", flush=True)

    if rank > 0:
        if "worker2" in hostname:
            config.data_dir = "./data"
        else:
            # Dùng /var/tmp thay vì /tmp để dữ liệu không bị xoá khi sập/reset máy ảo
            config.data_dir = "/var/tmp/fl_data"

    if config.download:
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        if local_rank == 0:
            if "worker2" in hostname:
                dl_domain = "svhn"
            elif "slave1" in hostname:
                dl_domain = "mnist"
            else:
                dl_domain = "usps"
            make_loader(config, dl_domain, train=True)
        comm.Barrier()

    if rank == 0:
        output = run_async_server(config, comm, device) if config.async_mode else run_sync_server(config, comm, device)
        print(f"MPI log written to {output}")
    elif config.async_mode:
        run_async_client(config, comm, rank, device)
    else:
        run_sync_client(config, comm, rank, device)


if __name__ == "__main__":
    main()
