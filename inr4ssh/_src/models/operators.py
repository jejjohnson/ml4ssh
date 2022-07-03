import torch
from typing import Callable


def mod_additive(x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    return x + mod


def mod_multiplicative(x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    return x * mod


def get_mod_operator(operator: str = "additive") -> Callable:

    if operator == "additive":
        return lambda x, mod: x + mod
    elif operator == "multiplicative":
        return lambda x, mod: x * mod
    else:
        raise ValueError(f"Unrecognized operator: {operator}")
