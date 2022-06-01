import equinox as eqx
import jax
import jax.numpy as jnp
Array = jnp.ndarray

class ReLU(eqx.Module):

    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)

class Tanh(eqx.Module):

    def __call__(self, x: Array) -> Array:
        return jax.nn.tanh(x)
    
class Swish(eqx.Module):
    beta: float = eqx.static_field()
    
    def __init__(self, beta: float=1.0):
        self.beta = beta

    def __call__(self, x: Array) -> Array:
        return x * jax.nn.sigmoid(self.beta * x)


class Sine(eqx.Module):
    """Sine Activation Function"""
    w0: Array = eqx.static_field()

    def __init__(self, w0: float=1.0):
        super().__init__()
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)


def get_activation(activation: str="relu", **kwargs):
    if activation == "identity":
        return eqx.nn.Identity()
    elif activation == "relu":
        return ReLU()
    elif activation == "tanh":
        return Tanh()
    elif activation == "swish":
        return Swish(beta=kwargs.get("beta", 1.0))
    elif activation == "sine":
        return Sine(w0=kwargs.get("w0", 1.0))
    else:
        raise ValueError(f"Unrecognized activation: {activation}")