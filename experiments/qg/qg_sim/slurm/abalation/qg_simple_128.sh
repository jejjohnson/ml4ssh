#!/bin/bash

# Low Resolution
sbatch --job-name=qg128 myscript.sh False 1e-4 100 10000 "128x128" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_128x128.nc"
sbatch --job-name=qg128 myscript.sh True 1e-6 100 10000 "128x128" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_128x128.nc"
sbatch --job-name=qg128 myscript.sh True 1e-4 100 10000 "128x128" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_128x128.nc"
sbatch --job-name=qg128 myscript.sh True 1e-2 100 10000 "128x128" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_128x128.nc"
sbatch --job-name=qg128 myscript.sh True 1e0 100 10000 "128x128" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_128x128.nc"

# Medium Resolution
sbatch --job-name=qg256 myscript.sh False 1e-4 50 5000 "256x256" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_256x256.nc"
sbatch --job-name=qg256 myscript.sh True 1e-6 50 5000 "256x256" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_256x256.nc"
sbatch --job-name=qg256 myscript.sh True 1e-4 50 5000 "256x256" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_256x256.nc"
sbatch --job-name=qg256 myscript.sh True 1e-2 50 5000 "256x256" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_256x256.nc"
sbatch --job-name=qg256 myscript.sh True 1e0 50 5000 "256x256" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_256x256.nc"

# High Resolution
sbatch --job-name=qg512 myscript.sh False 1e-4 10 2000 "512x512" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_512x512.nc"
sbatch --job-name=qg512 myscript.sh True 1e-6 10 2000 "512x512" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_512x512.nc"
sbatch --job-name=qg512 myscript.sh True 1e-4 10 2000 "512x512" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_512x512.nc"
sbatch --job-name=qg512 myscript.sh True 1e-2 10 2000 "512x512" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_512x512.nc"
sbatch --job-name=qg512 myscript.sh True 1e0 10 2000 "512x512" "/gpfswork/rech/cli/uvo53rl/data/qg_sim/qgsim_simple_512x512.nc"
