from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rl.q.agent import QLearningAgent, play_one_game_self_play
from tictactoe.ai import choose_move
from tictactoe.game import Player, TicTacToeGame


@dataclass
class EvalStats:
    win: int = 0
    loss: int = 0
    draw: int = 0


def default_save_path() -> Path:
    # Keep artifacts next to the agent package (rl/q/q_table.json)
    return Path(__file__).resolve().parent / "q_table.json"


def eval_vs_opponent(
    agent: QLearningAgent,
    *,
    games: int = 200,
    agent_player: Player = Player.X,
    opponent: str = "random",
    opponent_mistake_prob: float = 0.0,
    seed: Optional[int] = None,
) -> EvalStats:
    rng = random.Random(seed)
    stats = EvalStats()

    for _ in range(games):
        g = TicTacToeGame(starting_player=Player.X)
        while not g.done:
            if g.current_player == agent_player:
                a = agent.choose_action(g, train=False)
                g.step((a // 3, a % 3))
            else:
                if opponent == "random":
                    mv = rng.choice(g.legal_moves())
                elif opponent == "minimax":
                    mv = choose_move(g, mistake_prob=opponent_mistake_prob, rng=rng)
                else:
                    raise ValueError("opponent must be: random|minimax")
                g.step(mv)

        if g.winner is None:
            stats.draw += 1
        elif g.winner == agent_player:
            stats.win += 1
        else:
            stats.loss += 1

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Q-learning agent on Tic-Tac-Toe.")
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=str(default_save_path()))
    ap.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print a short progress line every N episodes (set 0 to disable).",
    )
    ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    agent_x = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, rng=rng)
    agent_o = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, rng=rng)

    def eps_at(ep: int) -> float:
        return max(0.05, args.epsilon * (0.99995**ep))

    print(f"training Q-learning for {args.episodes} episodes…")
    for ep in range(1, args.episodes + 1):
        agent_x.epsilon = eps_at(ep)
        agent_o.epsilon = eps_at(ep)

        starting = Player.X if (ep % 2 == 0) else Player.O
        play_one_game_self_play(agent_x, agent_o, train=True, starting_player=starting)

        if args.log_every > 0 and ep % args.log_every == 0:
            print(f"ep={ep} eps={agent_x.epsilon:.3f}")

        if args.eval_every > 0 and ep % args.eval_every == 0:
            s1 = eval_vs_opponent(agent_x, games=200, agent_player=Player.X, opponent="random", seed=args.seed)
            s2 = eval_vs_opponent(agent_x, games=200, agent_player=Player.O, opponent="random", seed=args.seed + 1)
            print(
                f"ep={ep} eps={agent_x.epsilon:.3f} | "
                f"as X vs random W/L/D={s1.win}/{s1.loss}/{s1.draw} | "
                f"as O vs random W/L/D={s2.win}/{s2.loss}/{s2.draw}"
            )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent_x.save(str(save_path))
    print(f"saved: {save_path}")

    s3 = eval_vs_opponent(
        agent_x,
        games=200,
        agent_player=Player.X,
        opponent="minimax",
        opponent_mistake_prob=0.2,
        seed=args.seed + 2,
    )
    print(f"final vs minimax(20% mistakes) as X W/L/D={s3.win}/{s3.loss}/{s3.draw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

