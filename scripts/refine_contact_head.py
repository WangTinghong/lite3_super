"""Fit only the contact head on existing data; freeze dynamics and normalization."""
from pathlib import Path
import argparse
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from scripts.fit_contact_world_model import evaluate
from lite3_supermimic.locomotion.contact_data import ContactDataset, check_splits, file_hash
from lite3_supermimic.locomotion.contact_model import ContactWorldModel, labels
from lite3_supermimic.supermimic import state as st


def freeze_dynamics(model):
    model.eval()
    model.requires_grad_(False)
    model.contact_head.requires_grad_(True)


def head_loss(model, dataset, ids, device):
    b = dataset.batch(ids, device)
    s, pose = b["s0"][:, 0], b["pose0"][:, 0]
    loss = 0
    for j in range(dataset.horizon):
        with torch.no_grad():
            relief = dataset.relief_at(ids, pose, device)
            features = model.net[:-1](model.in_norm(model._inputs(
                s, b["action"][:, j], b["prev"][:, j], b["context"][:, j], relief)))
            y = model.out_norm.unnormalize(model.net[-1](features))
            s = s + y[:, :st.STATE_DIM]
            pose = st.apply_pose_delta(pose, y[:, st.STATE_DIM:])
        loss = loss + F.binary_cross_entropy_with_logits(model.contact_head(features), labels(b["contact"][:, j]))
    return loss / dataset.horizon


def selection_score(report):
    # Same importance to each leg and each reported horizon. No test tuning.
    values = [group["all"]["contacts"][f"shank_only_{leg}"]["average_precision"]
              for group in report.values() for leg in st.LEGS]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError("Validation data has no shank-only positives")
    return sum(values) / len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="audit/contact_phase1/fit/contact_world_model.pt")
    p.add_argument("--output-dir", default="audit/contact_phase1/head_refine")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--eval-interval", type=int, default=250)
    a = p.parse_args()
    if min(a.updates, a.batch, a.eval_interval) < 1 or not 0 < a.lr < 1:
        p.error("Invalid updates, batch, evaluation interval, or learning rate")
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error("output-dir must be empty")
    source = torch.load(a.source, map_location="cpu", weights_only=False)
    if source.get("schema") != "lite3-contact-wm-offline-v1":
        p.error("Expected offline contact model")
    cfg = source["args"]
    train, val, test = [ContactDataset(cfg[key], cfg["horizon"]) for key in ("train", "validation", "test")]
    check_splits(train, val, test)
    if any(m["metadata"]["checkpoint_sha256"] != source["source_sha256"] for ds in (train, val, test) for m in ds.manifests):
        p.error("Dataset source checkpoint mismatch")
    torch.manual_seed(cfg["seed"])
    model = ContactWorldModel(tuple(source["wm_hidden"])).to(a.device)
    model.load_state_dict(source["world_model"])
    freeze_dynamics(model)
    optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=a.lr)
    # Reproduce the original held-out windows; test is a repeated regression set.
    gen = torch.Generator().manual_seed(cfg["seed"])
    vi = torch.randperm(len(val.index), generator=gen)[:cfg["eval_windows"]]
    ti = torch.randperm(len(test.index), generator=gen)[:cfg["eval_windows"]]
    train_gen = torch.Generator().manual_seed(cfg["seed"] + 100)
    before_val = evaluate(model, val, vi, a.device)
    best_score, best_it = selection_score(before_val), 0
    best_head = copy.deepcopy(model.contact_head.state_dict())
    out.mkdir(parents=True, exist_ok=True)
    for it in range(1, a.updates + 1):
        # Uniform windows, unweighted BCE: no class reweighting/threshold trick.
        ids = torch.randint(len(train.index), (a.batch,), generator=train_gen)
        loss = head_loss(model, train, ids, a.device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite contact loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.contact_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if it == 1 or it % 100 == 0:
            print(f"head {it}/{a.updates} BCE={float(loss.detach()):.5g}; dynamics frozen", flush=True)
        if it % a.eval_interval == 0 or it == a.updates:
            report = evaluate(model, val, vi, a.device)
            score = selection_score(report)
            print(f"validation mean shank-only AP={score:.5g} (best={best_score:.5g})", flush=True)
            (out / f"validation_{it}.json").write_text(json.dumps(report, indent=2) + "\n")
            if score > best_score:
                best_head, best_score, best_it = copy.deepcopy(model.contact_head.state_dict()), score, it
    model.contact_head.load_state_dict(best_head)
    for name, value in model.state_dict().items():
        if not name.startswith("contact_head.") and not torch.equal(value.cpu(), source["world_model"][name]):
            raise RuntimeError(f"Frozen dynamics changed: {name}")
    candidate_test = evaluate(model, test, ti, a.device)
    model.contact_head.load_state_dict({k.removeprefix("contact_head."): v for k, v in source["world_model"].items() if k.startswith("contact_head.")})
    before_test = evaluate(model, test, ti, a.device)
    model.contact_head.load_state_dict(best_head)
    artifact = dict(source)
    artifact.update(world_model={k: v.cpu() for k, v in model.state_dict().items()},
                    head_refinement=dict(args=vars(a), selected_update=best_it, source_hash=file_hash(a.source)))
    torch.save(artifact, out / "contact_world_model.pt")
    summary = dict(actor_updated=False, dynamics_unchanged=True, selected_head_update=best_it,
                   selection="validation mean AP over four shank-only heads and one/five steps",
                   test_note="Repeated regression set; confirm on a fresh seed before policy training", comparison={})
    for step, groups in candidate_test.items():
        summary["comparison"][step] = dict(
            before_fr=before_test[step]["all"]["contacts"]["shank_only_FR"],
            after_fr=groups["all"]["contacts"]["shank_only_FR"],
            dynamics_mse_before=before_test[step]["all"]["normalized_mse"],
            dynamics_mse_after=groups["all"]["normalized_mse"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(dict(before_test=before_test, after_test=candidate_test), indent=2) + "\n")
    print(f"Saved {out / 'summary.json'}; selected head update {best_it}, unchanged dynamics verified.", flush=True)


if __name__ == "__main__":
    main()
