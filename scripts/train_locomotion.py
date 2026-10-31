"""Train one of four independent Lite3 locomotion stages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lite3_supermimic.locomotion.train import main

if __name__ == "__main__":
    main()
