import sys, os
from pyprojroot import here

root = here(project_files=[".root"])
sys.path.append(str(root))




def add_model_args(parser):
    parser.add_argument('--model', type=str, default="siren")

    # NEURAL NETWORK SPECIFIC
    parser.add_argument('--out-dim', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-hidden', type=int, default=6)
    parser.add_argument('--model-seed', type=str, default=42)
    parser.add_argument('--activation', type=str, default="relu")

    # SIREN SPECIFIC
    parser.add_argument('--w0-initial', type=float, default=30.0)
    parser.add_argument('--w0', type=float, default=1.0)
    parser.add_argument('--final-scale', type=float, default=1.0)

    return parser



def get_model(config):
    """

    Args:
        model (str, optional): _description_. Defaults to "siren".
    """
    import jax.random as jrandom
    from ml4ssh._src.models_jax.siren import SirenNet
    from ml4ssh._src.models_jax.activations import get_activation
    from ml4ssh._src.models_jax.mlp import MLPNet
    init_key = jrandom.PRNGKey(config.model_seed)

    if config.model == "siren":
        model = SirenNet(
            in_dim=config.in_dim,
            out_dim=config.out_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            w0=config.w0,
            w0_initial=config.w0_initial,
            final_scale=config.final_scale,
            key=init_key
        )

    elif config.model == "mlp":

        activation = get_activation(config.activation)

        model = MLPNet(
            in_dim=config.in_dim,
            out_dim=config.out_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            key=init_key,
            activation=activation
        )

    else:
        raise ValueError(f"Unrecognized model: {config.model}")

    return model
