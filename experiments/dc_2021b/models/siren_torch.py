def add_model_args(parser):
    parser.add_argument('--model', type=str, default="siren")

    # NEURAL NETWORK SPECIFIC
    parser.add_argument('--out-dim', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-hidden', type=int, default=6)
    parser.add_argument('--model-seed', type=str, default=42)
    parser.add_argument('--activation', type=str, default="sine")
    parser.add_argument('--resnet', type=bool, default=False)

    # SIREN SPECIFIC
    parser.add_argument('--w0-initial', type=float, default=30.0)
    parser.add_argument('--w0', type=float, default=1.0)
    parser.add_argument('--final-scale', type=float, default=1.0)

    return parser