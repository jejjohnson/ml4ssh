def add_logger_args(parser):
    # Logger Params
     
    parser.add_argument('--project', type=str, default="gpflow4ssh")
    parser.add_argument('--entity', type=str, default="ige")
    parser.add_argument('--log-dir', type=str, default="./")
    parser.add_argument('--wandb-resume', type=str, default="allow")
    parser.add_argument('--wandb-mode', type=str, default="offline")
    parser.add_argument('--smoke-test', action="store_true")
    parser.add_argument('--id', type=str, default=None)
    return parser