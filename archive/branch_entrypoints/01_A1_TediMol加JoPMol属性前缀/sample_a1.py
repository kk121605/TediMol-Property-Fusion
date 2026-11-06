"""A1 sampling entry with explicit model-mode check."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_共用第一阶段底座与训练工具"))
from sample_compare import main
if __name__ == "__main__":
    main(expected_mode="a1_prefix")
