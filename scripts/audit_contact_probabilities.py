"""Read-only probability audit of a saved offline contact model. No fitting."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.fit_contact_world_model import evaluate
from lite3_supermimic.locomotion.contact_data import ContactDataset, check_splits
from lite3_supermimic.locomotion.contact_model import ContactWorldModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="audit/contact_phase1/fit/contact_world_model.pt")
    p.add_argument("--output", default="audit/contact_phase1/fit/probability_audit.json")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        p.error("output already exists; use a different --output path")
    saved = torch.load(a.model, map_location="cpu", weights_only=False)
    if saved.get("schema") != "lite3-contact-wm-offline-v1":
        p.error("Expected offline contact world-model artifact")
    cfg = saved["args"]
    model = ContactWorldModel(tuple(saved["wm_hidden"])).to(a.device)
    model.load_state_dict(saved["world_model"])
    val = ContactDataset(cfg["validation"], cfg["horizon"])
    test = ContactDataset(cfg["test"], cfg["horizon"])
    check_splits(val, test)
    for ds in (val, test):
        if any(m["metadata"]["checkpoint_sha256"] != saved["source_sha256"] for m in ds.manifests):
            p.error("Dataset checkpoint differs from model source")
    gen = torch.Generator().manual_seed(cfg["seed"])
    result = dict(actor_updated=False, model_updated=False, threshold=.5, splits={})
    for name, ds in (("validation", val), ("test", test)):
        ids = torch.randperm(len(ds.index), generator=gen)[:cfg["eval_windows"]]
        print(f"Auditing {name}: {len(ids)} windows", flush=True)
        scores = evaluate(model, ds, ids, a.device)
        result["splits"][name] = {step: {group: values.get("contacts", {})
                                               for group, values in groups.items()}
                                  for step, groups in scores.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    concise = {split: {step: groups["all"]["shank_only_FR"] for step, groups in steps.items()}
               for split, steps in result["splits"].items()}
    print(json.dumps(concise, indent=2))
    print(f"Full audit saved to {path}; thresholds/model unchanged.")


if __name__ == "__main__":
    main()
