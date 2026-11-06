"""Explicit registry for three physically separate fusion implementations."""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "a1_prefix": ("01_A1_TediMol加JoPMol属性前缀", "a1_fusion_model", "TediMolJoPMolPrefix"),
    "a2_rt_numeric": ("02_A2_TediMol加RegressionTransformer数值编码", "a2_fusion_model", "TediMolRTNumericalPrefix"),
    "a3_adaln_zero": ("03_A3_TediMol加GraphDiT启发逐层调节", "a3_fusion_model", "TediMolGraphDiTAdapter"),
}

def get_model_class(mode):
    folder, module, name = METHODS[mode]
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
    return getattr(importlib.import_module(module), name)

def fusion_sources():
    return [path for folder, _, _ in METHODS.values() for path in sorted((ROOT / folder).glob("*.py"))]
