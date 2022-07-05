# grab arguments
save_dir=$1

file_server=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/dc2022b_q/qg_sim.nc

wget --directory-prefix=$save_dir $file_server

