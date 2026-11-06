"""Bounded transition recording for offline contact/dynamics experiments.

Time and environment axes are retained; windows must never cross reset or file
boundaries. A manifest is published only when collection completes successfully.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SCHEMA = "lite3-contact-data-v1"
CONTACT_NAMES = [f"{kind}_{leg}" for kind in ("foot", "shank")
                 for leg in ("FR", "FL", "HR", "HL")]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ContactRecorder:
    def __init__(self, directory, metadata, chunk_steps=250):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise ValueError(f"Collection directory must be empty: {self.path}")
        self.metadata = metadata
        self.chunk_steps = chunk_steps
        self.rows, self.files = [], []
        self.count = 0

    def add(self, **values):
        self.rows.append({k: v.detach().cpu().clone() for k, v in values.items()})
        self.count += len(values["s0"])
        if len(self.rows) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f"chunk_{len(self.files):05d}.pt"
        torch.save({k: torch.stack([row[k] for row in self.rows])
                    for k in self.rows[0]}, self.path / name)
        self.files.append(name)
        self.rows.clear()
        print(f"contact data: {self.count} transitions saved", flush=True)

    def finish(self, terrain_audit):
        self.flush()
        manifest = dict(schema=SCHEMA, contact_names=CONTACT_NAMES,
                        metadata=self.metadata, files=self.files,
                        transitions=self.count, terrain_audit=terrain_audit)
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def window_starts(done, horizon):
    """All nonterminal windows; omit terminal transitions conservatively."""
    if len(done) < horizon:
        return torch.empty((0, 2), dtype=torch.long)
    return (~done.unfold(0, horizon, 1).any(-1)).nonzero()


class ContactDataset:
    def __init__(self, directories, horizon=5):
        self.horizon, self.chunks, self.manifests = horizon, [], []
        self.fields = []
        starts = []
        for directory in directories:
            directory = Path(directory)
            m = json.loads((directory / "manifest.json").read_text())
            if m["schema"] != SCHEMA or m["contact_names"] != CONTACT_NAMES:
                raise ValueError("Incompatible contact dataset")
            self.manifests.append(m)
            from lite3_supermimic.target.terrain import HeightField
            field = HeightField.load(str(directory / "field.pt"), "cpu") if (directory / "field.pt").exists() else None
            for name in m["files"]:
                data = torch.load(directory / name, map_location="cpu", weights_only=True)
                ids = window_starts(data["done"], horizon)
                if not len(ids):
                    continue
                starts.append(torch.cat((torch.full((len(ids), 1), len(self.chunks)), ids), 1))
                self.chunks.append(data)
                self.fields.append(field)
        if not starts:
            raise ValueError("No valid nonterminal windows; collect longer trajectories")
        self.index = torch.cat(starts)
        # Oversample transitions involving contact switches or shank-only support.
        self.event = torch.cat([self._events(c)[ids[:, 0], ids[:, 1]]
                                for c in self.chunks
                                for ids in [window_starts(c["done"], horizon)]])

    def _events(self, c):
        event = ((c["contact"] != c["contact0"]).any(-1)
                 | (c["contact"][..., 4:] & ~c["contact"][..., :4]).any(-1))
        return event.unfold(0, self.horizon, 1).any(-1)

    def batch(self, ids, device):
        rows = self.index[ids]
        result = {}
        for cid in rows[:, 0].unique().tolist():
            mask = rows[:, 0] == cid
            times = rows[mask, 1, None] + torch.arange(self.horizon)
            envs = rows[mask, 2, None]
            for key, value in self.chunks[cid].items():
                selected = value[times, envs]
                if key not in result:
                    result[key] = torch.empty((len(rows), *selected.shape[1:]), dtype=value.dtype)
                result[key][mask] = selected
        return {key: value.to(device) for key, value in result.items()}

    def relief_at(self, ids, pose, device):
        from lite3_supermimic.target import terrain as tr
        rows = self.index[ids]
        output = torch.empty((len(rows), tr.SCAN_POINTS), device=device)
        for cid in rows[:, 0].unique().tolist():
            mask = (rows[:, 0] == cid).to(device)
            field = self.fields[cid]
            if field is not None and field.h.device != pose.device:
                field.h = field.h.to(device)
            output[mask] = tr.relief(pose[mask, :2], pose[mask, 2], field)
        return output


def check_splits(*datasets):
    seen = set()
    checkpoint_hashes = set()
    for ds in datasets:
        seeds = {m["metadata"]["seed"] for m in ds.manifests}
        if seen & seeds:
            raise ValueError("Train/validation/test must use disjoint simulation seeds")
        seen |= seeds
        checkpoint_hashes |= {m["metadata"]["checkpoint_sha256"] for m in ds.manifests}
    if len(checkpoint_hashes) != 1:
        raise ValueError("This experiment requires one fixed collection policy checkpoint")
