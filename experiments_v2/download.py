import hydra
import loguru
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".root", pythonpath=True)


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg):
    # ===============================
    # logger
    # ===============================
    loguru.logger.info("Starting Download Script...")
    hydra.utils.instantiate(cfg.data)


if __name__ == "__main__":
    main()
