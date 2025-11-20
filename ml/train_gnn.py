"""
训练所有性质的GNN模型

用法:
  python train_gnn.py    # 训练所有性质的GNN模型
"""
from core import train_all_properties

if __name__ == "__main__":
    print("开始训练所有性质的GNN模型...")
    print("=" * 50)
    results = train_all_properties("gnn")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"成功训练: {success_count}/{len(results)} 个模型")
    
    if any(v is None for v in results.values()):
        print("\n失败的模型:")
        for prop, path in results.items():
            if path is None:
                print(f"  - {prop}")

