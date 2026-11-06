"""Read-only probability audit of a saved offline contact model. No fitting."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.fit_contact_world_model import evaluate
from lite3_supermimic.locomotion.contact_data import ContactDataset, check_splits, file_hash
from lite3_supermimic.locomotion.contact_model import ContactWorldModel


def fresh_datasets(cfg, directories, source_hash):
    # Read metadata only: do not load all training transitions for this check.
    used_seeds = set()
    for split in ("train", "validation", "test"):
        for directory in cfg[split]:
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            used_seeds.add(manifest["metadata"]["seed"])
    datasets = []
    for i, directory in enumerate(directories):
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        meta = manifest["metadata"]
        if meta["seed"] in used_seeds:
            raise ValueError(f"Fresh audit requires an unused seed: {directory}")
        if meta["checkpoint_sha256"] != source_hash:
            raise ValueError(f"Dataset checkpoint differs from model source: {directory}")
        datasets.append((f"fresh_{i}_{Path(directory).name}",
                         ContactDataset([directory], cfg["horizon"])))
    return datasets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="audit/contact_phase1/fit/contact_world_model.pt")
    p.add_argument("--output", default="audit/contact_phase1/fit/probability_audit.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fresh-data", nargs="+", help="New-seed collections; report each scene separately")
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
    if a.fresh_data:
        datasets = fresh_datasets(cfg, a.fresh_data, saved["source_sha256"])
    else:
        val = ContactDataset(cfg["validation"], cfg["horizon"])
        test = ContactDataset(cfg["test"], cfg["horizon"])
        check_splits(val, test)
        datasets = [("validation", val), ("test", test)]
    for _, ds in datasets:
        if any(m["metadata"]["checkpoint_sha256"] != saved["source_sha256"] for m in ds.manifests):
            p.error("Dataset checkpoint differs from model source")
    gen = torch.Generator().manual_seed(cfg["seed"])
    result = dict(actor_updated=False, model_updated=False, threshold=.5,
                  fresh_seed_audit=bool(a.fresh_data), model_sha256=file_hash(a.model),
                  source_sha256=saved["source_sha256"], splits={}, provenance={}, scores={})
    for name, ds in datasets:
        result["provenance"][name] = ds.manifests
        ids = torch.randperm(len(ds.index), generator=gen)[:cfg["eval_windows"]]
        print(f"Auditing {name}: {len(ids)} windows", flush=True)
        scores = evaluate(model, ds, ids, a.device)
        result["scores"][name] = scores
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
