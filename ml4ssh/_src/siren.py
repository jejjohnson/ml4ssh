from typing import Callable, List
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

Array = jnp.ndarray

class Sine(eqx.Module):
    """Sine Activation Function"""
    w0: Array = eqx.static_field()

    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)

    
class Siren(eqx.Module):
    """Siren Layer"""
    weight: Array
    bias: Array
    w0: Array = eqx.static_field()
    activation: eqx.Module

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        key: Array,  
        w0: float=1., 
        c: float=6.,
        activation=None
    ):
        super().__init__()
        w_key, b_key = jrandom.split(key)
        if w0 is None:
            # First layer
            w_max = 1 / in_dim
            b_max = 1 / jnp.sqrt(in_dim)
        else:
            w_max = jnp.sqrt(c / in_dim) / w0
            b_max = 1 / jnp.sqrt(in_dim) / w0
        self.weight = jrandom.uniform(
            key, (out_dim, in_dim), minval=-w_max, maxval=w_max
        )
        self.bias = jrandom.uniform(key, (out_dim,), minval=-b_max, maxval=b_max)
        self.w0 = w0
        self.activation = Sine(w0) if activation is None else activation

    def __call__(self, x: Array) -> Array:
        x = self.weight @ x + self.bias
        x = self.activation(x)
        return x


class SirenNet(eqx.Module):
    """SirenNet"""
    layers: List[Siren]
    w0: Array = eqx.static_field()
    final_scale: Array = eqx.static_field()
    final_activation: Callable[[Array], Array]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int,
        key: Array,
        w0_initial: float=30,
        w0: float=1.0,
        c: float=6.0,
        final_scale: float=1.0,
        final_activation: Callable[[Array], Array] = eqx.nn.Identity()
    ):
        super().__init__()
        keys = jrandom.split(key, n_hidden + 2)
        
        # First layer
        self.layers = [
            Siren(
                in_dim, hidden_dim, w0=w0_initial, c=c, key=keys[0], activation=None
            )
        ]
        
        # Hidden layers
        for ikey in keys[1:-1]:
            self.layers.append(
                Siren(
                    hidden_dim, hidden_dim, w0=w0, c=c, key=ikey, activation=None
                )
            )
        # Last layer
        self.layers.append(
            Siren(
                hidden_dim, out_dim, key=keys[-1], w0=w0, c=c, activation=final_activation
            )
        )

        self.w0 = w0
        self.final_scale = final_scale
        self.final_activation = final_activation

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return self.final_activation(x * self.final_scale)