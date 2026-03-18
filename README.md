## Tic-Tac-Toe (GUI)

Run the GUI game:

```bash
python gui_tictactoe.py
```

Notes:
- Uses Tkinter (included with most Python installs on Windows).
- Click a square to place your mark. Use **New Game** to reset.
- In the GUI you can pick the opponent (minimax / minimax with mistakes / RL if trained).

## Game manager (no UI)

The rules/state live in `tictactoe/game.py` as `TicTacToeGame`.

## Perfect AI opponent

`tictactoe/ai.py` implements perfect play (minimax). Example:

```python
from tictactoe.game import TicTacToeGame, Player
from tictactoe.ai import best_move

g = TicTacToeGame(starting_player=Player.X)
mv = best_move(g).move
g.step(mv)
```

## AI with mistakes

Use `choose_move()` to make the AI occasionally blunder:

```python
from tictactoe.game import TicTacToeGame, Player
from tictactoe.ai import choose_move

g = TicTacToeGame(starting_player=Player.O)
mv = choose_move(g, mistake_prob=0.2)  # 20% chance to pick a random non-best move
g.step(mv)
```

## RL (Q-learning) training

Train a simple tabular Q-learning agent by self-play:

```bash
python -m rl.q.train --episodes 30000
```

## RL (PyTorch DQN) training

Train a neural-network RL agent (Deep Q-Network) and save `rl/dqn/dqn_model.pt`:

```bash
python -m rl.dqn.train --episodes 60000
```


