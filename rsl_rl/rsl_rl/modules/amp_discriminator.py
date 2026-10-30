import torch
import torch.nn as nn
from torch import autograd


class AMPObservationNormalizer(nn.Module):
    def __init__(self, observation_dim: int, epsilon: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip = clip
        self.register_buffer("mean", torch.zeros(observation_dim))
        self.register_buffer("variance", torch.ones(observation_dim))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.shape[0] == 0:
            return
        values = values.detach().to(self.mean.device)
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        first_moment = self.variance * self.count
        second_moment = batch_variance * batch_count
        combined = (
            first_moment
            + second_moment
            + delta.square() * self.count * batch_count / total_count
        )
        self.mean.copy_(new_mean)
        self.variance.copy_(combined / total_count)
        self.count.copy_(total_count)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.variance + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)


class AMPDiscriminator(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims):
        super().__init__()
        if not hidden_dims:
            raise ValueError("AMP discriminator requires at least one hidden layer")

        layers = []
        input_dim = observation_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        features = torch.cat((state, next_state), dim=-1)
        return self.output(self.trunk(features))

    @torch.no_grad()
    def predict_reward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        prediction = self(state, next_state).squeeze(-1)
        return torch.clamp(1.0 - 0.25 * (prediction - 1.0).square(), min=0.0)

    def gradient_penalty(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, coefficient: float
    ) -> torch.Tensor:
        expert_pair = torch.cat((expert_state, expert_next_state), dim=-1).detach()
        expert_pair.requires_grad_(True)
        prediction = self.output(self.trunk(expert_pair))
        gradient = autograd.grad(
            outputs=prediction,
            inputs=expert_pair,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return coefficient * gradient.square().sum(dim=-1).mean()

