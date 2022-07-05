import subprocess
import yaml
import argparse
from pathlib import Path
from loguru import logger

def main(args):
    logger.info(f"Opening credentials file...")
    with open(f"{args.credentials_file}", "r") as file:
        credentials = yaml.full_load(file)
        print(credentials)

    logger.info(f"Extracting credentials...")
    username = credentials["aviso"]["username"]
    password = credentials["aviso"]["password"]

    logger.info(f"Checking save exists...")
    save_dir = Path(args.save_dir)
    assert save_dir.is_dir()

    logger.info(f"Checking bash script exists...")
    bash_script = Path("dl_dc21a.sh")
    cwd = Path.cwd()
    bash_script = cwd.joinpath(bash_script)
    assert bash_script.is_file()
    command = f"{bash_script} {username} {password} {save_dir}"

    logger.info(f"Executing commands...")
    rc = subprocess.call(
        f"{command} {username} {password} {save_dir}",
        shell=True
    )
    return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--credentials-file", type=str)
    args.add_argument("--save-dir", type=str, default="./")
    args = args.parse_args()
    main(args)