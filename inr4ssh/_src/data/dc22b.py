# # grab arguments
# save_dir=$1
#
# REF_EVAL=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022b_SSH_QG_mapping/dc_qg_eval.tar.gz
# wget --directory-prefix=$save_dir $REF_EVAL
# tar -xvf $save_dir/dc_qg_eval.tar.gz --directory=$save_dir
# rm -f $save_dir/dc_qg_eval.tar.gz
#
# REF_TRAIN=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022b_SSH_QG_mapping/dc_qg_train.tar.gz
# wget --directory-prefix=$save_dir $REF_TRAIN
# tar -xvf $save_dir/dc_qg_train.tar.gz --directory=$save_dir
# rm -f $save_dir/dc_qg_train.tar.gz
#
# OBS_FIELD=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022b_SSH_QG_mapping/dc_qg_obs_fullfields.tar.gz
# wget --directory-prefix=$save_dir $OBS_FIELD
# tar -xvf $save_dir/dc_qg_obs_fullfields.tar.gz --directory=$save_dir
# rm -f $save_dir/dc_qg_obs_fullfields.tar.gz
#
# OBS_JASON=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022b_SSH_QG_mapping/dc_qg_obs_jasonlike.tar.gz
# wget --directory-prefix=$save_dir $OBS_JASON
# tar -xvf $save_dir/dc_qg_obs_jasonlike.tar.gz --directory=$save_dir
# rm -f $save_dir/dc_qg_obs_jasonlike.tar.gz
#
# OBS_NADIR=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022b_SSH_QG_mapping/dc_qg_obs_nadirlike.tar.gz
# wget --directory-prefix=$save_dir $OBS_NADIR
# tar -xvf $save_dir/dc_qg_obs_nadirlike.tar.gz --directory=$save_dir
# rm -f $save_dir/dc_qg_obs_nadirlike.tar.gz
