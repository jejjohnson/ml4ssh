# grab arguments
save_dir=$1
url_alongtrack=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz


wget --directory-prefix=$save_dir $url_alongtrack
tar -xvf $save_dir/dc_obs.tar.gz --directory=$save_dir
rm -f $save_dir/dc_obs.tar.gz
