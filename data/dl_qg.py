import subprocess
import argparse
from pathlib import Path
from loguru import logger

def main(args):

    logger.info(f"Checking save exists...")
    save_dir = Path(args.save_dir)
    assert save_dir.is_dir()

    logger.info(f"Checking bash script exists...")
    bash_script = Path("dl_qg.sh")
    cwd = Path.cwd()
    bash_script = cwd.joinpath(bash_script)
    assert bash_script.is_file()
    command = f"{bash_script} {save_dir}"

    logger.info(f"Executing commands...")
    rc = subprocess.call(
        f"{command}",
        shell=True
    )
    logger.info(f"Done download...!")
    return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--save-dir", type=str, default="./")
    args = args.parse_args()
    main(args)