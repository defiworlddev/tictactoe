"""Deep Q-Network (PyTorch) agent package."""

from .agent import (
    DQN,
    DQNAgent,
    ReplayItem,
    action_to_move,
    encode_for_player,
    legal_actions,
    move_to_action,
)

__all__ = [
    "DQN",
    "DQNAgent",
    "ReplayItem",
    "action_to_move",
    "encode_for_player",
    "legal_actions",
    "move_to_action",
]

