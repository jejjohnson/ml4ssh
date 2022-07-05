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


    bash_script_traintest = Path("dl_dc21a.sh")
    bash_script_results = Path("dl_dc21a_results.sh")
    cwd = Path.cwd()
    bash_script_traintest = cwd.joinpath(bash_script_traintest)
    bash_script_results = cwd.joinpath(bash_script_results)

    logger.info(f"Checking bash script exists...")
    assert bash_script_traintest.is_file()
    assert bash_script_results.is_file()
    bash_script_args = f"{username} {password} {save_dir}"


    if args.traintest:
        logger.info(f"Executing traintest download...")
        rc = subprocess.call(
            f"{bash_script_traintest} {bash_script_args}",
            shell=True
        )
        logger.info(f"Done download...!")
    #
    if args.results:
        logger.info(f"Executing traintest download...")
        rc = subprocess.call(
            f"{bash_script_results} {bash_script_args}",
            shell=True
        )
        logger.info(f"Done download...!")
    return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--credentials-file", type=str)
    args.add_argument("--save-dir", type=str, default="./")
    args.add_argument("--traintest", action="store_true")
    args.add_argument("--results", action="store_true")
    args = args.parse_args()
    main(args)