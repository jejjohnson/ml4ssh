# grab arguments
save_dir=$1
URL_OBS=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz
URL_REF=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz

wget --directory-prefix=$save_dir URL_OBS
tar -xvf $save_dir/dc_obs.tar.gz --directory=$save_dir
rm -f $save_dir/dc_obs.tar.gz

wget --directory-prefix=$save_dir URL_REF
tar -xvf $save_dir/dc_ref.tar.gz --directory=$save_dir
rm -f $save_dir/dc_ref.tar.gz
