import json
from core import predict

def test_single_smiles_baseline():
    out = predict("CCO", model="baseline")
    assert "prediction" in out and "uncertainty" in out and "atom_importances" in out
    assert isinstance(out["atom_importances"], list)

def test_single_smiles_gnn_fallback_ok():
    # GNN may fallback to baseline if deps missing
    out = predict("CCO", model="gnn")
    assert "prediction" in out and "uncertainty" in out
