from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import SirenNet, ModulatedSirenNet
from inr4ssh._src.models.mfn import FourierNet, GaborNet

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
            use_bias=siren_config.use_bias,
            c=siren_config.c,
            final_activation=get_activation(siren_config.final_activation)
        )
    elif model == "modsiren":
        modsiren_config = config.modsiren
        return ModulatedSirenNet(
            dim_in=dim_in,
            dim_hidden=modsiren_config.hidden_dim,
            dim_out=dim_out,
            num_layers=modsiren_config.num_layers,
            w0=modsiren_config.w0,
            w0_initial=modsiren_config.w0_initial,
            c=modsiren_config.c,
            final_activation=get_activation(modsiren_config.final_activation),
            latent_dim=modsiren_config.latent_dim,
            num_layers_latent=modsiren_config.num_layers,
            operation=modsiren_config.operation
        )

    elif model == "fouriernet":
        fn_config = config.mfn
        return FourierNet(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=fn_config.hidden_dim,
            num_layers=fn_config.num_layers,
            input_scale=fn_config.input_scale,
            weight_scale=fn_config.weight_scale,
            use_bias=fn_config.use_bias,
            final_activation=get_activation(fn_config.final_activation)
        )
    elif model == "gabornet":
        fn_config = config.mfn
        return GaborNet(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=fn_config.hidden_dim,
            num_layers=fn_config.num_layers,
            input_scale=fn_config.input_scale,
            weight_scale=fn_config.weight_scale,
            alpha=fn_config.alpha,
            beta=fn_config.beta,
            use_bias=fn_config.use_bias,
            final_activation=get_activation(fn_config.final_activation)
        )
    else:
        raise ValueError(f"Unrecognized model: {model}")