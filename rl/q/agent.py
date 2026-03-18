from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tictactoe.game import Move, Player, TicTacToeGame

State = Tuple[int, ...]  # 9 ints from TicTacToeGame.encode()
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


@dataclass
class QLearningAgent:
    """
    Tabular Q-learning with epsilon-greedy action selection.

    Q is stored as: Q[state_str][action_str] = value
    (string keys keep JSON save/load simple)
    """

    alpha: float = 0.3
    gamma: float = 0.98
    epsilon: float = 0.2
    rng: random.Random = field(default_factory=random.Random)
    q: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def _skey(self, state: State) -> str:
        return ",".join(map(str, state))

    def _aget(self, state: State, action: Action) -> float:
        return self.q.get(self._skey(state), {}).get(str(action), 0.0)

    def _aset(self, state: State, action: Action, value: float) -> None:
        sk = self._skey(state)
        if sk not in self.q:
            self.q[sk] = {}
        self.q[sk][str(action)] = float(value)

    def choose_action(self, game: TicTacToeGame, train: bool = True) -> Action:
        acts = legal_actions(game)
        if not acts:
            raise ValueError("No legal actions.")

        if train and self.rng.random() < self.epsilon:
            return self.rng.choice(acts)

        state = game.encode()
        # Greedy with random tie-break.
        best_v: Optional[float] = None
        best: List[Action] = []
        for a in acts:
            v = self._aget(state, a)
            if best_v is None or v > best_v:
                best_v = v
                best = [a]
            elif v == best_v:
                best.append(a)
        return self.rng.choice(best)

    def learn(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: Optional[State],
        next_legal_actions: Optional[List[Action]],
        done: bool,
    ) -> None:
        q_sa = self._aget(state, action)
        if done or next_state is None or not next_legal_actions:
            target = reward
        else:
            next_best = max(self._aget(next_state, a) for a in next_legal_actions)
            target = reward + self.gamma * next_best
        self._aset(state, action, q_sa + self.alpha * (target - q_sa))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon, "q": self.q},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent = cls(
            alpha=float(data.get("alpha", 0.3)),
            gamma=float(data.get("gamma", 0.98)),
            epsilon=float(data.get("epsilon", 0.2)),
        )
        agent.q = {str(k): {str(ak): float(av) for ak, av in v.items()} for k, v in data["q"].items()}
        return agent


def play_one_game_self_play(
    agent_x: QLearningAgent,
    agent_o: QLearningAgent,
    *,
    train: bool = True,
    starting_player: Player = Player.X,
) -> Optional[Player]:
    """
    Returns winner (Player.X / Player.O) or None for draw.

    Learning is done with a simple terminal reward:
    - winner gets +1
    - loser gets -1
    - draw gets 0
    """
    g = TicTacToeGame(starting_player=starting_player)
    # track last (s,a) for each side to update on terminal
    last_state: Dict[Player, State] = {}
    last_action: Dict[Player, Action] = {}

    while not g.done:
        p = g.current_player
        agent = agent_x if p == Player.X else agent_o

        s = g.encode()
        a = agent.choose_action(g, train=train)
        mv = action_to_move(a)
        g.step(mv)

        last_state[p] = s
        last_action[p] = a

    # terminal update
    w = g.winner
    if train:
        for p in (Player.X, Player.O):
            if p not in last_state:
                continue
            if w is None:
                r = 0.0
            elif w == p:
                r = 1.0
            else:
                r = -1.0
            agent = agent_x if p == Player.X else agent_o
            agent.learn(
                state=last_state[p],
                action=last_action[p],
                reward=r,
                next_state=None,
                next_legal_actions=None,
                done=True,
            )
    return w

