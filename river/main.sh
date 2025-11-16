# preprocess
cd analysis/preprocess
if [ ! -f GSE211692_RAW.tar ]; then
    wget https://www.ncbi.nlm.nih.gov/geo/download/\?acc\=GSE211692\&format\=file -O GSE211692_RAW.tar
else
    echo "GSE211692_RAW.tar already exists, skipping download."
fi
bash run_preprocess.sh -1 GSE211692_RAW.tar work_dir preprocessed_dir mapping_file.txt

# upload to remote project
cd ../../
pwd
mkdir -p $outdir
cp analysis/preprocess/work_dir/full_data.csv $outdir

