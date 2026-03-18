from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from tictactoe.game import Move, Player, TicTacToeGame
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)

State = Tuple[int, ...]  # 9 ints row-major
Action = int  # 0..8 (row-major)


def move_to_action(move: Move) -> Action:
    r, c = move
    return r * 3 + c


def action_to_move(action: Action) -> Move:
    if not (0 <= action <= 8):
        raise ValueError("action must be in [0..8]")
    return action // 3, action % 3


def legal_actions(game: TicTacToeGame) -> List[Action]:
    return [move_to_action(m) for m in game.legal_moves()]


def encode_for_player(game: TicTacToeGame, player: Player) -> State:
    """
    Perspective encoding:
    - 1  = squares occupied by `player`
    - -1 = squares occupied by opponent
    - 0  = empty
    """
    s = game.encode()  # X=1, O=-1
    if player == Player.X:
        return s
    return tuple(-v for v in s)


class DQN(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: int = 64, out_dim: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReplayItem:
    s: torch.Tensor  # (9,)
    a: int
    r: float
    s2: Optional[torch.Tensor]  # (9,)
    done: bool
    next_legal_mask: Optional[torch.Tensor]  # (9,) bool, for s2


@dataclass
class DQNAgent:
    gamma: float = 0.99
    epsilon: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    min_replay: int = 2_000
    target_update_every: int = 500
    device: str = "cpu"
    rng: random.Random = field(default_factory=random.Random)

    model: DQN = field(init=False)
    target: DQN = field(init=False)
    opt: torch.optim.Optimizer = field(init=False)
    replay: List[ReplayItem] = field(default_factory=list)
    _learn_steps: int = 0

    def __post_init__(self) -> None:
        self.model = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _tensor_state(self, s: State) -> torch.Tensor:
        return torch.tensor(s, dtype=torch.float32, device=self.device)

    def choose_action(self, game: TicTacToeGame, *, for_player: Player, train: bool = True) -> Action:
        acts = legal_actions(game)
        if not acts:
            raise ValueError("No legal actions.")

        if train and self.rng.random() < self.epsilon:
            return self.rng.choice(acts)

        s = self._tensor_state(encode_for_player(game, for_player)).unsqueeze(0)  # (1,9)
        with torch.no_grad():
            q = self.model(s).squeeze(0)  # (9,)
            mask = torch.full((9,), False, dtype=torch.bool, device=self.device)
            mask[acts] = True
            q_masked = torch.where(mask, q, torch.tensor(-1e9, device=self.device))
            a = int(torch.argmax(q_masked).item())
        return a

    def remember(
        self,
        s: State,
        a: Action,
        r: float,
        s2: Optional[State],
        done: bool,
        next_legal_actions: Optional[List[Action]],
    ) -> None:
        s_t = self._tensor_state(s)
        s2_t = self._tensor_state(s2) if s2 is not None else None
        if s2_t is not None and next_legal_actions is not None:
            m = torch.zeros((9,), dtype=torch.bool, device=self.device)
            m[next_legal_actions] = True
        else:
            m = None
        self.replay.append(
            ReplayItem(s=s_t, a=int(a), r=float(r), s2=s2_t, done=bool(done), next_legal_mask=m)
        )
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

    def learn_step(self) -> Optional[float]:
        if len(self.replay) < self.min_replay:
            return None
        if len(self.replay) < self.batch_size:
            return None

        batch = self.rng.sample(self.replay, self.batch_size)
        s = torch.stack([b.s for b in batch], dim=0)  # (B,9)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64, device=self.device)  # (B,)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)  # (B,)
        done = torch.tensor([b.done for b in batch], dtype=torch.bool, device=self.device)  # (B,)

        q_sa = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            q_next = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
            for i, b in enumerate(batch):
                if b.done or b.s2 is None or b.next_legal_mask is None:
                    q_next[i] = 0.0
                    continue
                q2 = self.target(b.s2.unsqueeze(0)).squeeze(0)  # (9,)
                q2_masked = torch.where(b.next_legal_mask, q2, torch.tensor(-1e9, device=self.device))
                q_next[i] = torch.max(q2_masked).item()

            target = r + (~done).float() * (self.gamma * q_next)

        loss = torch.mean((q_sa - target) ** 2)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.model.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "gamma": self.gamma,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: str = "cpu",
    ) -> "DQNAgent":
        ckpt = torch.load(path, map_location=device)
        agent = cls(device=device)
        agent.model.load_state_dict(ckpt["model_state"])
        agent.target.load_state_dict(ckpt["model_state"])
        agent.model.eval()
        agent.target.eval()
        agent.gamma = float(ckpt.get("gamma", agent.gamma))
        agent.epsilon = 0.0
        return agent

