"""Contact-supervised dynamics with the original SuperMimic continuous outputs.

No contact truth is required at inference. Extra heads share the dynamics
trunk. The four shank-only events are learned directly, not as independent
products of marginal contact probabilities.
"""
from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from lite3_supermimic.supermimic.world_model import WorldModel
from lite3_supermimic.supermimic import state as st


class ContactWorldModel(WorldModel):
    def __init__(self, hidden=(1024, 1024, 1024)):
        super().__init__(hidden)
        self.contact_head = nn.Linear(hidden[-1], 12)

    def predict(self, s, a, prev, xi, relief):
        features = self.in_norm(self._inputs(s, a, prev, xi, relief))
        features = self.net[:-1](features)
        y = self.out_norm.unnormalize(self.net[-1](features))
        return (s + y[:, :st.STATE_DIM], y[:, st.STATE_DIM:], self.contact_head(features))

    def forward(self, s, a, prev, xi, relief):
        s1, delta, _ = self.predict(s, a, prev, xi, relief)
        return s1, delta

    def load_baseline(self, weights):
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if set(missing) != {"contact_head.weight", "contact_head.bias"} or unexpected:
            raise ValueError(f"Incompatible baseline: missing={missing}, unexpected={unexpected}")


def labels(contact):
    return torch.cat((contact, contact[..., 4:] & ~contact[..., :4]), -1).float()


def initialize_dynamics(world_model, optimizer, artifact_path, source_checkpoint):
    """Install contact-supervised continuous dynamics, leaving the head offline."""
    from .contact_data import file_hash
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if (artifact.get("schema") != "lite3-contact-wm-offline-v1"
        or artifact.get("source_sha256") != file_hash(source_checkpoint)):
        raise ValueError("Contact world model does not match the resumed policy checkpoint")
    dynamics = {key: value for key, value in artifact["world_model"].items()
                if not key.startswith("contact_head.")}
    if set(dynamics) != set(world_model.state_dict()):
        raise ValueError("Contact world model dynamics architecture differs")
    world_model.load_state_dict(dynamics, strict=True)
    optimizer.state.clear()  # Adam moments from the source model are stale.
    return Path(artifact_path)


def probability_diagnostics(prob, truth):
    """Threshold-free AP with tied scores grouped, plus calibration diagnostics."""
    order = prob.argsort(descending=True)
    score, y = prob[order], truth[order].double()
    ends = torch.cat(((score[:-1] != score[1:]).nonzero().flatten(),
                      torch.tensor([len(score) - 1], device=score.device)))
    tp = y.cumsum(0)[ends]
    precision = tp / (ends + 1)
    positive_count = int(truth.sum())
    ap = float((precision * torch.diff(tp, prepend=tp.new_zeros(1))).sum() / positive_count) if positive_count else None
    def distribution(mask):
        v = prob[mask]
        return dict(count=len(v), mean=float(v.mean()),
                    q10_q50_q90=[float(x) for x in v.quantile(v.new_tensor([.1, .5, .9]))]) if len(v) else dict(count=0)
    prevalence = float(truth.float().mean())
    return dict(average_precision=ap, prevalence=prevalence,
                constant_prevalence_brier=prevalence * (1 - prevalence),
                positive_probability=distribution(truth), negative_probability=distribution(~truth))


def classification_metrics(prob, target):
    pred, truth = prob >= .5, target.bool()
    tp = (pred & truth).sum(0).double()
    fp = (pred & ~truth).sum(0).double()
    fn = (~pred & truth).sum(0).double()
    names = [f"{kind}_{leg}" for kind in ("foot", "shank", "shank_only")
             for leg in ("FR", "FL", "HR", "HL")]
    def ratio(a, b):
        return float(a / b) if b else None
    return {name: dict(samples=len(prob), positives=int(truth[:, i].sum()),
                       precision=ratio(tp[i], tp[i] + fp[i]),
                       recall=ratio(tp[i], tp[i] + fn[i]),
                       false_positive_rate=ratio(fp[i], (~truth[:, i]).sum()),
                       brier=float((prob[:, i] - target[:, i]).square().mean()),
                       **probability_diagnostics(prob[:, i], truth[:, i]))
            for i, name in enumerate(names)}
