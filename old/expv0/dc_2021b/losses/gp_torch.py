import gpytorch

def add_loss_args(parser):
    parser.add_argument("--loss", type=str, default="elbo")
    parser.add_argument("--loss-beta", type=float, default=0.1)
    parser.add_argument("--loss-gamma", type=float, default=1.03)
    return parser


def get_loss_fn(likelihood, model, num_data, args):
    if args.loss == "elbo":
        loss = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=num_data, beta=args.loss_beta
        )
    elif args.loss == "pll":
        loss = gpytorch.mlls.PredictiveLogLikelihood(
            likelihood, model, num_data=num_data, beta=args.loss_beta
        )
    elif args.loss == "grelbo":
        loss = gpytorch.mlls.GammaRobustVariationalELBO(
            likelihood, model, gamma=args.loss_gamma
        )
    else:
        raise ValueError(f"Unrecognized loss: {loss}")
            
    return loss