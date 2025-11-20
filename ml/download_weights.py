# 创建小型演示权重文件
from core import train_baseline_demo, train_gnn_demo

if __name__ == "__main__":
    path_b = train_baseline_demo()
    print(f"[ok] baseline -> {path_b}")
    try:
        path_g = train_gnn_demo()
        print(f"[ok] gnn -> {path_g}")
    except Exception as e:
        print(f"[warn] gnn skipped: {e}")
