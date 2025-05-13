import os, time
import torch
import numpy as np
from torch.cuda.amp import autocast

from lightgcn_model import LightGCN
from din_model    import DIN
from ranknet_model import RankNet

# 全局配置
WARMUP_ITERS = 10
BENCH_ITERS  = 50

def file_size_mb(path):
    """返回文件大小（MB）"""
    return os.path.getsize(path) / 1024**2

def measure_latency_throughput(model, dummy_input, device):
    """
    测量模型在混合精度下的延迟和吞吐率
    Returns: median, p95, p99 latency (ms) and FPS
    """
    model = model.to(device).eval()
    # warm-up
    for _ in range(WARMUP_ITERS):
        with autocast():
            _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.time()
        with autocast():
            _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    lat_ms = np.array(times) * 1000
    median = np.percentile(lat_ms, 50)
    p95    = np.percentile(lat_ms, 95)
    p99    = np.percentile(lat_ms, 99)
    fps    = BENCH_ITERS / np.sum(times)
    return median, p95, p99, fps

def evaluate_autocast(model_cls, ckpt_path, dummy_input, name):
    print(f"\n=== {name}: Autocast Mixed Precision Report ===")
    # 模型大小
    size_mb = file_size_mb(ckpt_path)
    print(f"Model Size             : {size_mb:.2f} MB")

    # 加载模型
    device = torch.device("cuda")
    model = model_cls().to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    # 处理嵌套 state_dict
    if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
        key = "state_dict" if "state_dict" in state else "model_state_dict"
        state = state[key]
    model.load_state_dict(state)

    # 测量
    median, p95, p99, fps = measure_latency_throughput(model, dummy_input, device)
    print(f"Inference Latency (median)   : {median:.2f} ms")
    print(f"Inference Latency (95th pct) : {p95:.2f} ms")
    print(f"Inference Latency (99th pct) : {p99:.2f} ms")
    print(f"Inference Throughput         : {fps:.2f} FPS")

if __name__ == "__main__":
    device = torch.device("cuda")

    # LightGCN dummy input: edge_index [2, num_edges]
    num_nodes = 9622 + 5000
    dummy_lgc = torch.randint(0, num_nodes, (2, 8), device=device)
    evaluate_autocast(
        lambda: LightGCN(
            num_users=9622,
            num_items=5000,
            embedding_dim=128,
            num_layers=3,
            dropout=0.1
        ),
        "lightgcn.pth", dummy_lgc, "LightGCN"
    )

    # DIN dummy input: (candidate, history, length)
    dummy_c = torch.randn(16, 3, device=device)
    dummy_h = torch.randn(16, 10, 3, device=device)
    dummy_l = torch.randint(1, 11, (16,), device=device)
    evaluate_autocast(
        lambda: DIN(
            item_feat_dim=3,
            embedding_dim=128,
            attention_dim=128,
            mlp_hidden_dims=[128,64,32],
            dropout=0.2
        ),
        "din.pth", (dummy_c, dummy_h, dummy_l), "DIN"
    )

    # RankNet dummy input: (user, pos, neg)
    dummy_u = torch.randn(16, 128, device=device)
    dummy_p = torch.randn(16, 3, device=device)
    #dummy_n = torch.randn(16, 3, device=device)
    evaluate_autocast(
        lambda: RankNet(
            user_feat_dim=128,
            item_feat_dim=3,
            embedding_dim=128,
            hidden_dims=[128,64,32],
            dropout=0.2
        ),
        "ranknet.pth", (dummy_u, dummy_p), "RankNet"
    )
