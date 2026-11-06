"""A1-specific entry. Training is deferred until compute/protocol is approved."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_共用第一阶段底座与训练工具"))
from train_compare import main
if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "a1_prefix"]
    raise SystemExit(main(expected_mode="a1_prefix"))
