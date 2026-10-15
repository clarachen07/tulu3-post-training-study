"""Full frozen-asset preflight shared by both experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils.assets import (
    run_preflight as run_preflight,
    validate_stage_assets as validate_stage_assets,
    main,
)


if __name__ == "__main__":
    main()
