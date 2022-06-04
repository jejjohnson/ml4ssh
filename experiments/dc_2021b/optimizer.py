

def add_optimizer_args(parser):
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--prefetch-buffer", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser

def add_optimizer_gpflow_args(parser):
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-ng", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--prefetch-buffer", type=int, default=5)
    return parser


def get_optimizer(config):
    import optax

    if config.optimizer == "adam":
        optimizer = optax.adam(config.learning_rate)

    elif config.optimizer == "sgd":
        optimizer = optax.sgd(config.learning_rate)
    else:
        raise ValueError(f"Unrecognized Optimizer: {config.optimizer}")

    return optimizer