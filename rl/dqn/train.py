from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rl.dqn.agent import DQNAgent, action_to_move, encode_for_player, legal_actions
from tictactoe.ai import choose_move
from tictactoe.game import Player, TicTacToeGame


@dataclass
class EvalStats:
    win: int = 0
    loss: int = 0
    draw: int = 0


def default_save_path() -> Path:
    # Keep artifacts next to the agent package (rl/dqn/dqn_model.pt)
    return Path(__file__).resolve().parent / "dqn_model.pt"


def play_episode_vs_opponent(
    agent: DQNAgent,
    *,
    agent_player: Player,
    opponent: str,
    opponent_mistake_prob: float,
    rng: random.Random,
    train: bool,
) -> Optional[Player]:
    """
    One episode where the agent plays as `agent_player` and the opponent plays the other side.
    Transition is defined over the agent's turns only: (s -> a -> env(opponent) -> s2).
    """
    g = TicTacToeGame(starting_player=Player.X)

    while not g.done:
        if g.current_player != agent_player:
            if opponent == "random":
                mv = rng.choice(g.legal_moves())
            elif opponent == "minimax":
                mv = choose_move(g, mistake_prob=opponent_mistake_prob, rng=rng)
            else:
                raise ValueError("opponent must be: random|minimax")
            g.step(mv)
            continue

        s = encode_for_player(g, agent_player)
        a = agent.choose_action(g, for_player=agent_player, train=train)
        g.step(action_to_move(a))

        if g.done:
            if g.winner is None:
                r = 0.0
            elif g.winner == agent_player:
                r = 1.0
            else:
                r = -1.0
            if train:
                agent.remember(s, a, r, None, True, None)
            break

        if opponent == "random":
            mv2 = rng.choice(g.legal_moves())
        elif opponent == "minimax":
            mv2 = choose_move(g, mistake_prob=opponent_mistake_prob, rng=rng)
        else:
            raise ValueError("opponent must be: random|minimax")
        g.step(mv2)

        if g.done:
            if g.winner is None:
                r = 0.0
            elif g.winner == agent_player:
                r = 1.0
            else:
                r = -1.0
            if train:
                agent.remember(s, a, r, None, True, None)
            break

        s2 = encode_for_player(g, agent_player)
        next_acts = legal_actions(g)
        if train:
            agent.remember(s, a, 0.0, s2, False, next_acts)

    return g.winner


@torch.no_grad()
def eval_vs_opponent(
    agent: DQNAgent,
    *,
    games: int,
    agent_player: Player,
    opponent: str,
    opponent_mistake_prob: float,
    seed: int,
) -> EvalStats:
    rng = random.Random(seed)
    stats = EvalStats()
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    try:
        for _ in range(games):
            w = play_episode_vs_opponent(
                agent,
                agent_player=agent_player,
                opponent=opponent,
                opponent_mistake_prob=opponent_mistake_prob,
                rng=rng,
                train=False,
            )
            if w is None:
                stats.draw += 1
            elif w == agent_player:
                stats.win += 1
            else:
                stats.loss += 1
    finally:
        agent.epsilon = old_eps
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a PyTorch DQN agent for Tic-Tac-Toe.")
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu|cuda")
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-replay", type=int, default=2000)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--target-update-every", type=int, default=500)
    ap.add_argument("--opponent", type=str, default="minimax", help="random|minimax")
    ap.add_argument("--opponent-mistake-prob", type=float, default=0)
    ap.add_argument("--save", type=str, default=str(default_save_path()))
    ap.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print a short progress line every N episodes (set 0 to disable).",
    )
    ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    agent = DQNAgent(
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr,
        batch_size=args.batch_size,
        min_replay=args.min_replay,
        replay_size=args.replay_size,
        target_update_every=args.target_update_every,
        device=args.device,
        rng=rng,
    )

    def eps_at(ep: int) -> float:
        return max(0.02, args.epsilon * (0.99995**ep))

    print(
        f"training DQN for {args.episodes} episodes on device={args.device} "
        f"(min_replay={args.min_replay}, batch={args.batch_size})…"
    )
    for ep in range(1, args.episodes + 1):
        agent.epsilon = eps_at(ep)
        agent_player = Player.X if (ep % 2 == 0) else Player.O

        play_episode_vs_opponent(
            agent,
            agent_player=agent_player,
            opponent=args.opponent,
            opponent_mistake_prob=args.opponent_mistake_prob,
            rng=rng,
            train=True,
        )
        loss = agent.learn_step()

        if args.log_every > 0 and ep % args.log_every == 0:
            if loss is None:
                print(f"ep={ep} eps={agent.epsilon:.3f} (warming up replay: {len(agent.replay)}/{args.min_replay})")
            else:
                print(f"ep={ep} eps={agent.epsilon:.3f} loss={loss:.4f}")

        if args.eval_every > 0 and ep % args.eval_every == 0:
            s1 = eval_vs_opponent(
                agent,
                games=200,
                agent_player=Player.X,
                opponent=args.opponent,
                opponent_mistake_prob=args.opponent_mistake_prob,
                seed=args.seed,
            )
            s2 = eval_vs_opponent(
                agent,
                games=200,
                agent_player=Player.O,
                opponent=args.opponent,
                opponent_mistake_prob=args.opponent_mistake_prob,
                seed=args.seed + 1,
            )
            msg = (
                f"ep={ep} eps={agent.epsilon:.3f}"
                f" | as X W/L/D={s1.win}/{s1.loss}/{s1.draw}"
                f" | as O W/L/D={s2.win}/{s2.loss}/{s2.draw}"
            )
            if loss is not None:
                msg += f" | loss={loss:.4f}"
            print(msg)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    print(f"saved: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

