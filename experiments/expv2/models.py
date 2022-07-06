from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import SirenNet

def model_factory(model, dim_in, dim_out, config):

    if model == "siren":
        siren_config = config.siren
        return SirenNet(
            dim_in=dim_in,
            dim_hidden=siren_config.hidden_dim,
            dim_out=dim_out,
            num_layers=siren_config.num_layers,
            w0=siren_config.w0,
            w0_initial=siren_config.w0_initial,
            c=siren_config.c,
            final_activation=get_activation(siren_config.final_activation)
        )
    else:
        raise ValueError(f"Unrecognized model: {model}")