import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def collect_results_to_csv(sweep_dir, drop_long_strings=False):
    if not sweep_dir.exists():
        logger.error("pt sweep data directory does not exist")
        raise FileNotFoundError("data dir doesn't exist")
    
    names = [dir for dir in sweep_dir.iterdir() if dir.is_dir()]

    data = []
    
    for name in names:
        instance_details_file = name / "instance_details.json"

        if not instance_details_file.exists():
            logger.warning(f"{name} does not exist")
            continue

        with open(instance_details_file, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
                if drop_long_strings:
                    obj.pop("h_str")
                    obj.pop("h_gamma_str")
                obj["observables_file"] = name / "pmrqmc_pt_observables.csv"
                obj["swaps_file"] = name / "pmrqmc_pt_swaps.csv"
                obj["rank_timing_file"] = name / "pmrqmc_pt_rank_timing.csv"
                obj["flow_file"] = name / "pmrqmc_pt_flow.csv"
                if obj["validate_final_run"]:
                    obj["validation_file"] = name / "validation.csv"
                data.append(obj)
            except(json.JSONDecodeError, OSError):
                logger.warning(f"json error parsing {name}")
                continue

    df = pd.DataFrame(data)

    out_file = sweep_dir / "pt_sweep.csv"
    df.to_csv(out_file)

    print(f"collected {len(data)} results to {out_file}")

    return df
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--drop-long-strings", action="store_true")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    drop_long_strings = args.drop_long_strings

    collect_results_to_csv(sweep_dir, drop_long_strings)
