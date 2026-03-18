import tkinter as tk
from tkinter import messagebox
from typing import Optional
import os
from pathlib import Path

from tictactoe.ai import choose_move
from tictactoe.game import Player, TicTacToeGame
from rl.dqn.agent import DQNAgent, action_to_move as dqn_action_to_move
from rl.q.agent import QLearningAgent, action_to_move as q_action_to_move


class TicTacToeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tic-Tac-Toe")
        self.resizable(False, False)

        self.game = TicTacToeGame(starting_player=Player.X)
        self.status_var = tk.StringVar(value=f"Turn: {self.game.current_player.value}")
        self.vs_ai = tk.BooleanVar(value=True)
        self.opponent = tk.StringVar(value="minimax_mistake")
        self.dqn_agent: Optional[DQNAgent] = self._try_load_dqn_agent()
        self.q_agent: Optional[QLearningAgent] = self._try_load_q_agent()
        self.buttons: list[list[tk.Button]] = []

        root = tk.Frame(self, padx=12, pady=12)
        root.grid(row=0, column=0)

        header = tk.Frame(root)
        header.grid(row=0, column=0, sticky="ew")

        tk.Label(header, textvariable=self.status_var, font=("Segoe UI", 12)).grid(
            row=0, column=0, sticky="w"
        )

        controls = tk.Frame(header)
        controls.grid(row=0, column=1, sticky="e")
        tk.Checkbutton(controls, text="Play vs AI (O)", variable=self.vs_ai).grid(
            row=0, column=0, padx=(0, 8)
        )
        tk.OptionMenu(
            controls,
            self.opponent,
            "minimax",
            "minimax_mistake",
            "rl",
        ).grid(row=0, column=1, padx=(0, 8))
        tk.Button(controls, text="New Game", command=self.new_game).grid(
            row=0, column=2
        )

        grid = tk.Frame(root, pady=10)
        grid.grid(row=1, column=0)

        for r in range(3):
            row_btns: list[tk.Button] = []
            for c in range(3):
                b = tk.Button(
                    grid,
                    text="",
                    width=6,
                    height=3,
                    font=("Segoe UI", 20, "bold"),
                    command=lambda rr=r, cc=c: self.on_click(rr, cc),
                )
                b.grid(row=r, column=c, padx=6, pady=6)
                row_btns.append(b)
            self.buttons.append(row_btns)

    def new_game(self) -> None:
        self.game.reset(starting_player=Player.X)
        self.status_var.set(f"Turn: {self.game.current_player.value}")
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="", state="normal")
        self.maybe_ai_move()

    def on_click(self, r: int, c: int) -> None:
        if self.vs_ai.get() and self.game.current_player != Player.X:
            return
        if not self.game.is_legal((r, c)):
            return
        result = self.game.step((r, c))
        self.buttons[r][c].config(text=result.player.value)

        if self._maybe_end(result.winner, result.done):
            return
        self.status_var.set(f"Turn: {self.game.current_player.value}")
        self.maybe_ai_move()

    def maybe_ai_move(self) -> None:
        if not self.vs_ai.get():
            return
        if self.game.done:
            return
        if self.game.current_player != Player.O:
            return
        self.after(80, self._do_ai_move)

    def _do_ai_move(self) -> None:
        if not self.vs_ai.get() or self.game.done or self.game.current_player != Player.O:
            return
        mv = self._pick_ai_move()
        result = self.game.step(mv)
        r, c = mv
        self.buttons[r][c].config(text=result.player.value)

        if self._maybe_end(result.winner, result.done):
            return
        self.status_var.set(f"Turn: {self.game.current_player.value}")

    def _maybe_end(self, winner: Optional[Player], done: bool) -> bool:
        if winner is not None:
            self.end_game(f"{winner.value} wins!")
            return True
        if done:
            self.end_game("Draw!")
            return True
        return False

    def end_game(self, msg: str) -> None:
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(state="disabled")
        messagebox.showinfo("Game Over", msg)

    def _try_load_dqn_agent(self) -> Optional[DQNAgent]:
        path = Path(__file__).resolve().parent / "rl" / "dqn" / "dqn_model.pt"
        try:
            agent = DQNAgent.load(str(path), device="cpu")
            agent.epsilon = 0.0
            return agent
        except Exception:
            return None

    def _try_load_q_agent(self) -> Optional[QLearningAgent]:
        path = Path(__file__).resolve().parent / "rl" / "q" / "q_table.json"
        try:
            agent = QLearningAgent.load(str(path))
            agent.epsilon = 0.0
            return agent
        except Exception:
            return None

    def _pick_ai_move(self) -> tuple[int, int]:
        kind = self.opponent.get()
        if kind == "minimax":
            return choose_move(self.game, mistake_prob=0.0)
        if kind == "minimax_mistake":
            return choose_move(self.game, mistake_prob=0.2)
        if kind == "rl":
            if self.dqn_agent is not None:
                a = self.dqn_agent.choose_action(self.game, for_player=Player.O, train=False)
                print(f"DQN agent chose action: {a}")
                return dqn_action_to_move(a)
            # if self.q_agent is not None:
            #     a = self.q_agent.choose_action(self.game, train=False)
            #     return q_action_to_move(a)
            return choose_move(self.game, mistake_prob=0.2)
        return choose_move(self.game, mistake_prob=0.2)


if __name__ == "__main__":
    TicTacToeApp().mainloop()

