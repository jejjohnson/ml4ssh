def add_optimizer_args(parser):
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-ng", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser