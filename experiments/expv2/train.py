import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # LOGGING
    parser.add_argument('--project', type=str, default=config.wandb_project)
    parser.add_argument('--entity', type=str, default=config.wandb_entity)
    parser.add_argument('--log-dir', type=str, default=config.wandb_log_dir)
    parser.add_argument('--wandb-resume', type=str, default=config.wandb_resume)
    parser.add_argument('--wandb-mode', type=str, default=config.wandb_mode)
    parser.add_argument('--smoke-test', action="store_true")
    parser.add_argument('--wandb-id', type=str, default=config.wandb_id)
    # DATA DIRECTORY
    parser.add_argument('--train-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/train")
    parser.add_argument('--ref-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/ref/")
    parser.add_argument('--test-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/test/")
    # DATA PREPROCESS
    # longitude subset
    parser.add_argument('--lon-min', type=float, default=285.0)
    parser.add_argument('--lon-max', type=float, default=315.0)
    parser.add_argument('--dlon', type=float, default=0.2)
    
    # latitude subset
    parser.add_argument('--lat-min', type=float, default=23.0)
    parser.add_argument('--lat-max', type=float, default=53.0)
    parser.add_argument('--dlat', type=float, default=0.2)
    
    # temporal subset
    parser.add_argument('--time-min', type=str, default="2016-12-01")
    parser.add_argument('--time-max', type=str, default="2018-01-31")
    parser.add_argument('--dtime', type=str, default="1_D")
    
    # Buffer Params
    parser.add_argument('--lon-buffer', type=float, default=1.0)
    parser.add_argument('--lat-buffer', type=float, default=1.0)
    parser.add_argument('--time-buffer', type=float, default=7.0)
    # =====================
    # FEATURES
    parser.add_argument("--julian-time", type=bool, default=True)
    parser.add_argument("--feature-scaler", type=str, default="minmax")
    # TRAIN/TEST SPLIT
    parser.add_argument("--train-size", type=float, default=0.90)
    parser.add_argument("--split-seed", type=int, default=666)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    # MODEL
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
    
    # LOSSES
    # =======
    parser.add_argument("--loss", type=str, default="mse")
    # OPTIMIZER
    # =========
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--prefetch-buffer", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    # DATA POSTPROCESS
    # ====================
    # longitude subset
    parser.add_argument('--eval-lon-min', type=float, default=295.0)
    parser.add_argument('--eval-lon-max', type=float, default=305.0)
    parser.add_argument('--eval-dlon', type=float, default=0.2)
    
    # latitude subset
    parser.add_argument('--eval-lat-min', type=float, default=33.0)
    parser.add_argument('--eval-lat-max', type=float, default=43.0)
    parser.add_argument('--eval-dlat', type=float, default=0.2)
    
    # temporal subset
    parser.add_argument('--eval-time-min', type=str, default="2017-01-01")
    parser.add_argument('--eval-time-max', type=str, default="2017-12-31")
    parser.add_argument('--eval-dtime', type=str, default="1_D")
    
    # OI params
    parser.add_argument('--eval-lon-buffer', type=float, default=2.0)
    parser.add_argument('--eval-lat-buffer', type=float, default=2.0)
    parser.add_argument('--eval-time-buffer', type=float, default=7.0)
    # data

    # VIZ
    # =====
    # longitude subset
    parser.add_argument('--viz-lon-min', type=float, default=295.0)
    parser.add_argument('--viz-lon-max', type=float, default=305.0)
    parser.add_argument('--viz-dlon', type=float, default=0.1)
    
    # latitude subset
    parser.add_argument('--viz-lat-min', type=float, default=33.0)
    parser.add_argument('--viz-lat-max', type=float, default=43.0)
    parser.add_argument('--viz-dlat', type=float, default=0.1)
    
    # temporal subset
    parser.add_argument('--viz-time-min', type=str, default="2017-01-01")
    parser.add_argument('--viz-time-max', type=str, default="2017-12-31")
    parser.add_argument('--viz-dtime', type=str, default="1_D")
    
    # OI params
    parser.add_argument('--viz-lon-buffer', type=float, default=1.0)
    parser.add_argument('--viz-lat-buffer', type=float, default=1.0)
    parser.add_argument('--viz-time-buffer', type=float, default=7.0)

    args = parser.parse_args()

    main(args)