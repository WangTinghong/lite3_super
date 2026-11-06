"""Contact-supervised dynamics with the original SuperMimic continuous outputs.

No contact truth is required at inference. Extra heads share the dynamics
trunk. The four shank-only events are learned directly, not as independent
products of marginal contact probabilities.
"""
from __future__ import annotations

import torch
from torch import nn

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
                       brier=float((prob[:, i] - target[:, i]).square().mean()))
            for i, name in enumerate(names)}
