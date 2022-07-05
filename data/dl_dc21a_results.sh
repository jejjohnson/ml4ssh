# grab arguments
dir=$3
username=$1
password=$2

# make results directory
result_dir=$dir/results

# DOWNLOAD OTHER RESULTS
wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_BASELINE.nc
wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_DUACS.nc
#wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_BFN.nc
#wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_MIOST.nc
#wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_DYMOST.nc
#wget --user $username --password $password --directory-prefix=$result_dir https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/OSE_ssh_mapping_4dvarNet.nc
