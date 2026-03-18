from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .game import Move, Player, TicTacToeGame


@dataclass(frozen=True)
class BestMove:
    move: Move
    score: int  # +1 win, 0 draw, -1 loss (from perspective of the "for_player")


def best_move(game: TicTacToeGame, for_player: Optional[Player] = None) -> BestMove:
    """
    Perfect-play move chooser (minimax).

    - If for_player is None, assumes the side to play (game.current_player).
    - Returns BestMove(move, score) where score is from for_player's perspective.
    """
    if game.done:
        raise ValueError("Game already finished.")

    player = for_player or game.current_player
    if player != game.current_player:
        raise ValueError("best_move expects for_player == game.current_player for this game state.")

    cache: Dict[Tuple[Tuple[int, ...], Player, Player], BestMove] = {}
    return _minimax(game, maximizing_player=player, cache=cache)


def choose_move(
    game: TicTacToeGame,
    mistake_prob: float = 0.2,
    rng: Optional[random.Random] = None,
) -> Move:
    """
    AI policy: plays optimally most of the time, but makes a mistake with probability
    `mistake_prob` by choosing a random *non-best* legal move when possible.
    """
    if not (0.0 <= mistake_prob <= 1.0):
        raise ValueError("mistake_prob must be in [0.0, 1.0]")
    if game.done:
        raise ValueError("Game already finished.")

    r = rng or random
    legal = game.legal_moves()
    if not legal:
        raise ValueError("No legal moves available.")

    best = best_move(game).move
    if r.random() >= mistake_prob:
        return best

    alternatives = [m for m in legal if m != best]
    if not alternatives:
        return best
    return r.choice(alternatives)


def _minimax(
    game: TicTacToeGame,
    maximizing_player: Player,
    cache: Dict[Tuple[Tuple[int, ...], Player, Player], BestMove],
) -> BestMove:
    key = (game.encode(), game.current_player, maximizing_player)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if game.done:
        if game.winner is None:
            out = BestMove(move=(-1, -1), score=0)
        elif game.winner == maximizing_player:
            out = BestMove(move=(-1, -1), score=1)
        else:
            out = BestMove(move=(-1, -1), score=-1)
        cache[key] = out
        return out

    is_max_turn = game.current_player == maximizing_player
    best: Optional[BestMove] = None

    for mv in game.legal_moves():
        nxt = game.clone()
        nxt.step(mv)
        child = _minimax(nxt, maximizing_player=maximizing_player, cache=cache)
        candidate = BestMove(move=mv, score=child.score)

        if best is None:
            best = candidate
            continue

        if is_max_turn:
            if candidate.score > best.score:
                best = candidate
        else:
            if candidate.score < best.score:
                best = candidate

        # Short-circuit if we found a forced outcome.
        if is_max_turn and best.score == 1:
            break
        if (not is_max_turn) and best.score == -1:
            break

    assert best is not None
    cache[key] = best
    return best

