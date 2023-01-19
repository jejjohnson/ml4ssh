gpus=${1:-1}
cores=${2:-16}
srun --account=yrf@v100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=$cores --gres=gpu:$gpus --hint=nomultithread -C v100-16g --qos=qos_gpu-dev --time=02:00:00 --pty bash
