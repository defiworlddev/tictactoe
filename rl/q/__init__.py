"""Q-learning (tabular) agent package."""

from .agent import QLearningAgent, action_to_move, legal_actions, move_to_action, play_one_game_self_play

__all__ = [
    "QLearningAgent",
    "action_to_move",
    "legal_actions",
    "move_to_action",
    "play_one_game_self_play",
]

