"""
命令行接口和可导入的预测函数封装。

用法示例:
 python inference.py --smiles "CCO" --model baseline --json
 python inference.py --csv data/logp.csv --output out.csv
 python inference.py --xlsx data/example.xlsx --output out.csv
"""
import sys, json, argparse, os, csv
from typing import Dict, Any
from core import predict, ensure_demo_dataset

# 主函数，处理命令行参数并执行单条预测或批量预测
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--model", type=str, default="gnn", choices=["gnn","baseline"])
    p.add_argument("--json", action="store_true", help="print JSON result")
    p.add_argument("--csv", type=str, default=None, help="input CSV with 'smiles' column")
    p.add_argument("--xlsx", type=str, default=None, help="input XLSX with 'smiles' column")
    p.add_argument("--output", type=str, default=None, help="CSV output path")
    a = p.parse_args()

    if a.smiles:
        out = predict(a.smiles, model=a.model)
        if a.json:
            print(json.dumps(out))
        else:
            print(out)
        sys.exit(0)

    if a.csv or a.xlsx:
        in_path = None
        data = []
        
        if a.csv:
            in_path = a.csv if os.path.isabs(a.csv) else os.path.join(os.path.dirname(__file__), a.csv)
            if not os.path.exists(in_path):
                print(f"Missing file: {in_path}", file=sys.stderr); sys.exit(1)
            with open(in_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                data = list(reader)
        elif a.xlsx:
            try:
                import pandas as pd
            except ImportError:
                print("XLSX支持需要pandas和openpyxl。请安装: pip install pandas openpyxl", file=sys.stderr)
                sys.exit(1)
            in_path = a.xlsx if os.path.isabs(a.xlsx) else os.path.join(os.path.dirname(__file__), a.xlsx)
            if not os.path.exists(in_path):
                print(f"Missing file: {in_path}", file=sys.stderr); sys.exit(1)
            df = pd.read_excel(in_path)
            data = df.to_dict("records")
        
        rows = []
        total = len(data) or 1
        for i, row in enumerate(data, 1):
            smi = row.get("smiles","")
            res: Dict[str, Any] = predict(smi, model=a.model)
            # 构建行数据，包含所有性质的预测
            row_data = {"smiles": smi}
            if res.get("properties"):
                # 多性质格式
                for prop_key, prop_data in res["properties"].items():
                    row_data[f"{prop_key}_prediction"] = prop_data.get("prediction", "")
                    row_data[f"{prop_key}_uncertainty"] = prop_data.get("uncertainty", "")
            else:
                # 向后兼容：单性质格式
                row_data["prediction"] = res.get("prediction", "")
                row_data["uncertainty"] = res.get("uncertainty", "")
            row_data["atom_importances_json"] = json.dumps(res.get("atom_importances", []))
            rows.append(row_data)
            print(f"PROGRESS {i}/{total}")
            sys.stdout.flush()
        outp = a.output or os.path.join(os.path.dirname(in_path), "results.csv")
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {outp}")
        sys.exit(0)

    p.print_help()

if __name__ == "__main__":
    main()
