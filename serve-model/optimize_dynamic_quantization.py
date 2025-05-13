import os
import time
import torch
import numpy as np

# 强制只用 CPU，屏蔽所有 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lightgcn_model import LightGCN
from din_model    import DIN
from ranknet_model import RankNet

# 全局配置
WARMUP_ITERS = 10
BENCH_ITERS  = 50

def file_size_mb(path):
    """返回文件大小（MB）"""
    return os.path.getsize(path) / 1024**2

class ModelEvaluator:
    def __init__(self, model_cls, ckpt_path, dummy_input, name):
        self.model_cls   = model_cls
        self.ckpt_path   = ckpt_path
        # 确保所有输入都在 CPU 上
        if isinstance(dummy_input, tuple):
            self.dummy_input = tuple(inp.cpu() if isinstance(inp, torch.Tensor) else inp
                                     for inp in dummy_input)
        else:
            inp = dummy_input
            self.dummy_input = inp.cpu() if isinstance(inp, torch.Tensor) else inp
        self.name        = name
        self.device      = torch.device("cpu")

    def load_model(self):
        model = self.model_cls().to(self.device).eval()
        state = torch.load(self.ckpt_path, map_location="cpu")
        # 处理可能的嵌套 state_dict
        if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
            key   = "state_dict" if "state_dict" in state else "model_state_dict"
            state = state[key]
        model.load_state_dict(state)
        return model

    def measure_times(self, model):
        """返回推理时间列表（秒）"""
        inp = self.dummy_input
        # warm-up
        for _ in range(WARMUP_ITERS):
            _ = model(*inp) if isinstance(inp, tuple) else model(inp)
        times = []
        for _ in range(BENCH_ITERS):
            t0 = time.time()
            _ = model(*inp) if isinstance(inp, tuple) else model(inp)
            times.append(time.time() - t0)
        return times

    def run(self):
        print(f"\n--- {self.name} (CPU Dynamic Quant) ---")
        # 原始模型评估
        orig_size  = file_size_mb(self.ckpt_path)
        model      = self.load_model()
        orig_times = self.measure_times(model)
        lat_ms     = np.array(orig_times) * 1000
        median     = np.percentile(lat_ms, 50)
        print(f"Original  | size {orig_size:.2f} MB | median latency {median:.2f} ms")

        # 动态量化 + JIT tracing
        base_model = self.load_model().to(self.device)
        q_model    = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        # 确保 tracing 输入在 CPU
        example_input = self.dummy_input
        traced = torch.jit.trace(
            q_model,
            example_input if isinstance(example_input, tuple) else (example_input,)
        )
        out_pt = f"{self.name}_opt_cpu.pt"
        traced.save(out_pt)

        opt_size   = file_size_mb(out_pt)
        opt_times  = self.measure_times(traced)
        lat_ms_opt = np.array(opt_times) * 1000
        median_opt = np.percentile(lat_ms_opt, 50)
        print(f"Optimized | size {opt_size:.2f} MB | median latency {median_opt:.2f} ms")


if __name__ == "__main__":
    # LightGCN Dummy Input
    num_users, num_items = 9622, 5000
    num_nodes           = num_users + num_items
    dummy_lc            = torch.randint(0, num_nodes, (2, 8), device="cpu")
    ModelEvaluator(
        lambda: LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=128,
            num_layers=3,
            dropout=0.1
        ),
        "lightgcn.pth",
        dummy_lc,
        "LightGCN"
    ).run()

    # DIN Dummy Input
    dummy_c = torch.randn(16, 3, device="cpu")    # item_feat_dim=3
    dummy_h = torch.randn(16, 10, 3, device="cpu")# seq_len=10
    dummy_l = torch.randint(1, 11, (16,), device="cpu")
    ModelEvaluator(
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
    ).run()

    # RankNet Dummy Input
    dummy_u = torch.randn(16, 128, device="cpu")  # user_feat_dim=128
    dummy_p = torch.randn(16, 3, device="cpu")    # item_feat_dim=3
    dummy_n = torch.randn(16, 3, device="cpu")
    ModelEvaluator(
        lambda: RankNet(
            user_feat_dim=128,
            item_feat_dim=3,
            embedding_dim=128,
            hidden_dims=[128,64,32],
            dropout=0.2
        ),
        "ranknet.pth",
        (dummy_u, dummy_p, dummy_n),
        "RankNet"
    ).run()
