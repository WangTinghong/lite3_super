"""Offline contact/dynamics fitting. Never updates an actor or optimizer from RL."""
from pathlib import Path
import argparse
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from lite3_supermimic.locomotion.contact_data import ContactDataset, check_splits, file_hash
from lite3_supermimic.locomotion.contact_model import ContactWorldModel, labels, classification_metrics
from lite3_supermimic.supermimic.world_model import WorldModel
from lite3_supermimic.supermimic import state as st


def evaluate(model, dataset, ids, device, batch_size=512):
    model.eval()
    records = {1: [], dataset.horizon: []}
    scale = model.out_norm.var.sqrt() + 1e-6
    with torch.no_grad():
        for selection in ids.split(batch_size):
            b = dataset.batch(selection, device)
            s, pose = b["s0"][:, 0], b["pose0"][:, 0]
            for j in range(dataset.horizon):
                # Free pose/terrain rollout, not recorded scans after step one.
                relief = dataset.relief_at(selection, pose, device)
                args = (s, b["action"][:, j], b["prev"][:, j], b["context"][:, j], relief)
                if isinstance(model, ContactWorldModel):
                    s, delta, logits = model.predict(*args)
                else:
                    s, delta = model(*args)
                    logits = None
                pose = st.apply_pose_delta(pose, delta)
                if j + 1 not in records:
                    continue
                error = torch.cat((s - b["s1"][:, j], delta - b["delta"][:, j]), -1)
                foot = (s[:, st.FEET] - b["s1"][:, j, st.FEET]).reshape(-1, 4, 3)
                near = b["near"][:, 0]
                event = (b["contact"][:, j] != b["contact0"][:, j]).any(-1)
                shank = (b["contact"][:, j, 4:] & ~b["contact"][:, j, :4]).any(-1)
                world_base_error = (s[:, 0] + dataset.ground_at(selection, pose, device)
                                    - b["s1"][:, j, 0] - dataset.ground_at(selection, b["pose1"][:, j], device))
                values = {
                    "base_world_z_sq": world_base_error.square(),
                    "foot_world_z_sq": (foot[..., 2] + world_base_error[:, None]).square().mean(-1),
                    "normalized_mse": (error / scale).square().mean(-1),
                    "vx_sq": (s[:, 4] - b["s1"][:, j, 4]).square(),
                    "yaw_rate_sq": (s[:, 9] - b["s1"][:, j, 9]).square(),
                    "foot_relative_z_sq": foot[..., 2].square().mean(-1),
                    "base_xy_sq": (pose[:, :2] - b["pose1"][:, j, :2]).square().sum(-1),
                    "near": near, "event": event, "shank": shank,
                    "target": labels(b["contact"][:, j]),
                }
                if logits is not None:
                    values["prob"] = logits.sigmoid()
                    values["bce"] = F.binary_cross_entropy_with_logits(logits, values["target"], reduction="none").mean(-1)
                records[j + 1].append({k: v.cpu() for k, v in values.items()})
    report = {}
    for horizon, batches in records.items():
        data = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
        groups = {}
        for name, mask in dict(all=torch.ones(len(ids), dtype=torch.bool),
                               near_riser=data["near"], contact_switch=data["event"],
                               shank_only=data["shank"]).items():
            count = int(mask.sum())
            group = {"samples": count}
            if count:
                group["normalized_mse"] = float(data["normalized_mse"][mask].mean())
                for key in ("vx_sq", "yaw_rate_sq", "foot_relative_z_sq", "base_xy_sq", "base_world_z_sq", "foot_world_z_sq"):
                    group[key.replace("_sq", "_rmse")] = float(data[key][mask].mean().sqrt())
                if "prob" in data:
                    group["bce"] = float(data["bce"][mask].mean())
                    group["contacts"] = classification_metrics(data["prob"][mask], data["target"][mask])
            groups[name] = group
        report[f"step_{horizon}"] = groups
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--contact-weight", type=float, default=.2)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.updates, args.batch, args.horizon, args.eval_interval, args.eval_windows) < 1 or not (0 < args.lr < 1) or not (0 <= args.contact_weight < 100):
        p.error("Invalid positive counts, learning rate, or contact weight")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("output-dir must be empty")
    torch.manual_seed(args.seed)
    datasets = [ContactDataset(paths, args.horizon) for paths in (args.train, args.validation, args.test)]
    train, val, test = datasets
    check_splits(*datasets)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_hash = file_hash(args.checkpoint)
    if source_hash != train.manifests[0]["metadata"]["checkpoint_sha256"]:
        p.error("checkpoint differs from the policy/world-model used to collect this dataset")
    hidden = tuple(ckpt["cfg"]["wm_hidden"])
    baseline = WorldModel(hidden).to(args.device)
    baseline.load_state_dict(ckpt["world_model"])
    model = ContactWorldModel(hidden).to(args.device)
    model.load_baseline(ckpt["world_model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    valid_ids = torch.randperm(len(val.index), generator=generator)[:args.eval_windows]
    test_ids = torch.randperm(len(test.index), generator=generator)[:args.eval_windows]
    print(f"windows: train={len(train.index)} validation={len(val.index)} test={len(test.index)}; actor frozen", flush=True)
    initial = evaluate(baseline, val, valid_ids, args.device)
    (out / "baseline_validation.json").write_text(json.dumps(initial, indent=2) + "\n")
    events = train.event.nonzero().flatten()
    best, best_weights, best_it = float("inf"), None, 0
    scale = model.out_norm.var.sqrt() + 1e-6  # Keep the baseline normalization fixed.
    for it in range(1, args.updates + 1):
        model.train()
        n_event = args.batch // 2 if len(events) else 0
        ids = torch.cat((torch.randint(len(train.index), (args.batch - n_event,), generator=generator),
                         events[torch.randint(len(events), (n_event,), generator=generator)] if n_event else events[:0]))
        b = train.batch(ids, args.device)
        s = b["s0"][:, 0]
        dynamics, contact_loss = 0, 0
        for j in range(args.horizon):
            s, delta, logits = model.predict(s, b["action"][:, j], b["prev"][:, j],
                                             b["context"][:, j], b["relief"][:, j])
            err = torch.cat((s - b["s1"][:, j], delta - b["delta"][:, j]), -1)
            dynamics = dynamics + (err / scale).square().mean()
            contact_loss = contact_loss + F.binary_cross_entropy_with_logits(logits, labels(b["contact"][:, j]))
        loss = (dynamics + args.contact_weight * contact_loss) / args.horizon
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite offline model loss; no actor/checkpoint was modified")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if it % 50 == 0 or it == 1:
            print(f"update {it}/{args.updates} dynamics={float(dynamics.detach())/args.horizon:.5g} contact={float(contact_loss.detach())/args.horizon:.5g}", flush=True)
        if it % args.eval_interval == 0 or it == args.updates:
            result = evaluate(model, val, valid_ids, args.device)
            aggregate = result[f"step_{args.horizon}"]["all"]
            score = aggregate["normalized_mse"] + args.contact_weight * aggregate["bce"]
            print(f"validation {it}: score={score:.5g}", flush=True)
            (out / f"validation_{it}.json").write_text(json.dumps(result, indent=2) + "\n")
            if score < best:
                best, best_it = score, it
                best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_weights)
    # An offline artifact, deliberately not a resumable actor-training checkpoint.
    torch.save(dict(schema="lite3-contact-wm-offline-v1", world_model=best_weights,
                    wm_hidden=hidden, source_sha256=source_hash, args=vars(args),
                    selected_update=best_it), out / "contact_world_model.pt")
    report = dict(source_sha256=source_hash, selected_update=best_it,
                  selection="validation full-distribution dynamics MSE + contact_weight * BCE",
                  train_windows=len(train.index), event_windows=len(events),
                  normalization="baseline frozen", actor_updated=False,
                  rollout="recorded actions, predicted pose and resampled terrain; terminal windows excluded",
                  baseline_test=evaluate(baseline, test, test_ids, args.device),
                  contact_model_test=evaluate(model, test, test_ids, args.device),
                  datasets={name: ds.manifests for name, ds in zip(("train", "validation", "test"), datasets)})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = dict(actor_updated=False, selected_update=best_it,
                   status="OFFLINE_VALIDATION_ONLY", comparisons={})
    for step in report["baseline_test"]:
        comparison = {}
        for group in report["baseline_test"][step]:
            old = report["baseline_test"][step][group]
            new = report["contact_model_test"][step][group]
            if not old["samples"]:
                comparison[group] = {"samples": 0}
                continue
            comparison[group] = dict(samples=old["samples"],
                baseline_normalized_mse=old["normalized_mse"],
                candidate_normalized_mse=new["normalized_mse"],
                dynamics_error_ratio=new["normalized_mse"] / max(old["normalized_mse"], 1e-12),
                fr_shank_only=new["contacts"]["shank_only_FR"])
        summary["comparisons"][step] = comparison
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {out / 'report.json'}; this is model validation, not proof of improved walking.", flush=True)


if __name__ == "__main__":
    main()
