import os, time
import torch
import numpy as np

from lightgcn_model import LightGCN
from din_model       import DIN
from ranknet_model   import RankNet

# 全局配置
WARMUP_ITERS = 10
BENCH_ITERS  = 50

def file_size_mb(path):
    return os.path.getsize(path) / 1024**2

def measure(model, dummy_input, device):
    model = model.to(device).eval()
    # Warm-up
    for _ in range(WARMUP_ITERS):
        _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Benchmark
    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.time()
        _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    lat_ms = np.array(times) * 1000
    median = np.percentile(lat_ms, 50)
    p95    = np.percentile(lat_ms, 95)
    fps    = BENCH_ITERS / np.sum(times)
    return median, p95, fps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LightGCN
    print("\n--- LightGCN Baseline ---")
    lc_size = file_size_mb("lightgcn.pth")
    lc_model = LightGCN(num_users=9622, num_items=5000, embedding_dim=128, num_layers=3, dropout=0.1)
    # dummy edge_index of small size for latency test
    dummy_lc = torch.randint(0, 9622+5000, (2, 8), device=device)
    lc_model.load_state_dict(torch.load("lightgcn.pth", map_location=device))
    lc_med, lc_p95, lc_fps = measure(lc_model, dummy_lc, device)
    print(f"File Size: {lc_size:.2f} MB")
    print(f"P50 Latency: {lc_med:.2f} ms, P95 Latency: {lc_p95:.2f} ms, Throughput: {lc_fps:.2f} FPS")

    # DIN
    print("\n--- DIN Baseline ---")
    din_size = file_size_mb("din.pth")
    din_model = DIN(item_feat_dim=3, embedding_dim=128, attention_dim=128, mlp_hidden_dims=[128,64,32], dropout=0.2)
    state = torch.load("din.pth", map_location=device)
    din_model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    # dummy inputs
    dummy_c = torch.randn(16, 3, device=device)
    dummy_h = torch.randn(16, 10, 3, device=device)
    dummy_l = torch.randint(1, 11, (16,), device=device)
    din_med, din_p95, din_fps = measure(din_model, (dummy_c, dummy_h, dummy_l), device)
    print(f"File Size: {din_size:.2f} MB")
    print(f"P50 Latency: {din_med:.2f} ms, P95 Latency: {din_p95:.2f} ms, Throughput: {din_fps:.2f} FPS")

    # RankNet
    print("\n--- RankNet Baseline ---")
    rn_size = file_size_mb("ranknet.pth")
    rn_model = RankNet(user_feat_dim=128, item_feat_dim=3, embedding_dim=128, hidden_dims=[128,64,32], dropout=0.2)
    state = torch.load("ranknet.pth", map_location=device)
    rn_model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    # dummy inputs
    dummy_u = torch.randn(16, 128, device=device)
    dummy_p = torch.randn(16, 3, device=device)
    dummy_n = torch.randn(16, 3, device=device)
    rn_med, rn_p95, rn_fps = measure(rn_model, (dummy_u, dummy_p, dummy_n), device)
    print(f"File Size: {rn_size:.2f} MB")
    print(f"P50 Latency: {rn_med:.2f} ms, P95 Latency: {rn_p95:.2f} ms, Throughput: {rn_fps:.2f} FPS")
