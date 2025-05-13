import os, time
import torch
import numpy as np

from lightgcn_model import LightGCN
from din_model    import DIN
from ranknet_model import RankNet

# 全局配置
WARMUP_ITERS = 10
BENCH_ITERS  = 50

def file_size_mb(path):
    return os.path.getsize(path) / 1024**2


def measure_times(model, dummy_input, device):
    """返回推理时间列表（秒），供后续计算延迟和吞吐"""
    model = model.to(device).eval()
    inp = dummy_input
    # warm-up
    for _ in range(WARMUP_ITERS):
        _ = model(*inp) if isinstance(inp, tuple) else model(inp)
    if device.type == 'cuda': torch.cuda.synchronize()

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.time()
        _ = model(*inp) if isinstance(inp, tuple) else model(inp)
        if device.type == 'cuda': torch.cuda.synchronize()
        times.append(time.time() - t0)
    return times


def evaluate_compile(model_cls, ckpt_path, dummy_input, name):
    print(f"\n=== {name}: Torch.compile Optimization Report ===")
    # 文件大小不变
    size_mb = file_size_mb(ckpt_path)
    print(f"Model Size: {size_mb:.2f} MB")

    # 加载原始模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls().to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
        key = "state_dict" if "state_dict" in state else "model_state_dict"
        state = state[key]
    model.load_state_dict(state)

    # 原始测速
    times_orig = measure_times(model, dummy_input, device)
    lat_ms_orig = np.array(times_orig) * 1000
    med_orig = np.percentile(lat_ms_orig, 50)
    p95_orig = np.percentile(lat_ms_orig, 95)
    p99_orig = np.percentile(lat_ms_orig, 99)
    fps_orig = BENCH_ITERS / np.sum(times_orig)
    print(f"Original Latency (median): {med_orig:.2f} ms, P95: {p95_orig:.2f} ms, P99: {p99_orig:.2f} ms")
    print(f"Original Throughput: {fps_orig:.2f} FPS")

    # Torch.compile 编译
    compiled = torch.compile(model)
    times_comp = measure_times(compiled, dummy_input, device)
    lat_ms_comp = np.array(times_comp) * 1000
    med_comp = np.percentile(lat_ms_comp, 50)
    p95_comp = np.percentile(lat_ms_comp, 95)
    p99_comp = np.percentile(lat_ms_comp, 99)
    fps_comp = BENCH_ITERS / np.sum(times_comp)
    print(f"Compiled Latency (median): {med_comp:.2f} ms, P95: {p95_comp:.2f} ms, P99: {p99_comp:.2f} ms")
    print(f"Compiled Throughput: {fps_comp:.2f} FPS")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LightGCN dummy input
    num_nodes = 9622 + 5000
    dummy_lc = torch.randint(0, num_nodes, (2, 8), device=device)
    evaluate_compile(
        lambda: LightGCN(
            num_users=9622,
            num_items=5000,
            embedding_dim=128,
            num_layers=3,
            dropout=0.1
        ),
        "lightgcn.pth",
        dummy_lc,
        "LightGCN"
    )

    # DIN dummy input
    dummy_c = torch.randn(16, 3, device=device)
    dummy_h = torch.randn(16, 10, 3, device=device)
    dummy_l = torch.randint(1, 11, (16,), device=device)
    evaluate_compile(
        lambda: DIN(
            item_feat_dim=3,
            embedding_dim=128,
            attention_dim=128,
            mlp_hidden_dims=[128,64,32],
            dropout=0.2
        ),
        "din.pth",
        (dummy_c, dummy_h, dummy_l),
        "DIN"
    )

    # RankNet dummy input
    dummy_u = torch.randn(16, 128, device=device)
    dummy_p = torch.randn(16, 3, device=device)
    #dummy_n = torch.randn(16, 3, device=device)
    evaluate_compile(
        lambda: RankNet(
            user_feat_dim=128,
            item_feat_dim=3,
            embedding_dim=128,
            hidden_dims=[128,64,32],
            dropout=0.2
        ),
        "ranknet.pth",
        (dummy_u, dummy_p),
        "RankNet"
    )
