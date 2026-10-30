import torch


class AMPReplayBuffer:
    def __init__(self, observation_dim: int, buffer_size: int, device: str):
        self.device = torch.device(device)
        self.buffer_size = int(buffer_size)
        self.states = torch.zeros(self.buffer_size, observation_dim, device=self.device)
        self.next_states = torch.zeros_like(self.states)
        self.write_index = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        if states.shape != next_states.shape:
            raise ValueError("AMP states and next_states must have the same shape")
        if states.shape[0] == 0:
            return

        states = states.detach()
        next_states = next_states.detach()
        if states.shape[0] >= self.buffer_size:
            states = states[-self.buffer_size:]
            next_states = next_states[-self.buffer_size:]

        count = states.shape[0]
        first_count = min(count, self.buffer_size - self.write_index)
        second_count = count - first_count
        self.states[self.write_index:self.write_index + first_count].copy_(states[:first_count])
        self.next_states[self.write_index:self.write_index + first_count].copy_(next_states[:first_count])
        if second_count:
            self.states[:second_count].copy_(states[first_count:])
            self.next_states[:second_count].copy_(next_states[first_count:])

        self.write_index = (self.write_index + count) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + count)

    def sample(self, batch_size: int):
        if self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty AMP replay buffer")
        indices = torch.randint(self.num_samples, (batch_size,), device=self.device)
        return self.states[indices], self.next_states[indices]

