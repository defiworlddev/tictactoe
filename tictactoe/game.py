from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class Player(str, Enum):
    X = "X"
    O = "O"

    def other(self) -> "Player":
        return Player.O if self == Player.X else Player.X


Move = Tuple[int, int]  # (row, col)


@dataclass
class StepResult:
    move: Move
    player: Player
    winner: Optional[Player]
    done: bool


class TicTacToeGame:
    """
    Pure game/state manager (no UI).
    - Board values are: None / Player.X / Player.O
    - Coordinates are 0-based (row, col), each in [0..2]
    """

    def __init__(self, starting_player: Player = Player.X) -> None:
        self._starting_player = starting_player
        self.reset()

    def reset(self, starting_player: Optional[Player] = None) -> None:
        if starting_player is not None:
            self._starting_player = starting_player
        self.board: List[List[Optional[Player]]] = [[None for _ in range(3)] for _ in range(3)]
        self.current_player: Player = self._starting_player
        self.winner: Optional[Player] = None
        self.done: bool = False

    def clone(self) -> "TicTacToeGame":
        g = TicTacToeGame(self._starting_player)
        g.board = [row[:] for row in self.board]
        g.current_player = self.current_player
        g.winner = self.winner
        g.done = self.done
        return g

    def legal_moves(self) -> List[Move]:
        if self.done:
            return []
        moves: List[Move] = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] is None:
                    moves.append((r, c))
        return moves

    def is_legal(self, move: Move) -> bool:
        r, c = move
        return (
            not self.done
            and 0 <= r < 3
            and 0 <= c < 3
            and self.board[r][c] is None
        )

    def step(self, move: Move) -> StepResult:
        if not self.is_legal(move):
            raise ValueError(f"Illegal move: {move}")

        r, c = move
        player = self.current_player
        self.board[r][c] = player

        w = self._compute_winner()
        self.winner = w
        if w is not None:
            self.done = True
        elif self._is_draw():
            self.done = True

        if not self.done:
            self.current_player = self.current_player.other()

        return StepResult(move=move, player=player, winner=self.winner, done=self.done)

    def outcome(self) -> Optional[Player]:
        """Returns winner if game is finished with a win; None otherwise (draw or ongoing)."""
        return self.winner

    def encode(self) -> Tuple[int, ...]:
        """
        RL-friendly encoding: 9 ints row-major.
        0 = empty, 1 = X, -1 = O
        """
        out: List[int] = []
        for r in range(3):
            for c in range(3):
                v = self.board[r][c]
                if v is None:
                    out.append(0)
                elif v == Player.X:
                    out.append(1)
                else:
                    out.append(-1)
        return tuple(out)

    def _is_draw(self) -> bool:
        return all(self.board[r][c] is not None for r in range(3) for c in range(3))

    def _compute_winner(self) -> Optional[Player]:
        b = self.board
        lines: Sequence[Sequence[Optional[Player]]] = (
            b[0],
            b[1],
            b[2],
            (b[0][0], b[1][0], b[2][0]),
            (b[0][1], b[1][1], b[2][1]),
            (b[0][2], b[1][2], b[2][2]),
            (b[0][0], b[1][1], b[2][2]),
            (b[0][2], b[1][1], b[2][0]),
        )
        for line in lines:
            a, b1, c = line
            if a is not None and a == b1 == c:
                return a
        return None

