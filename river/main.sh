eval "$(pixi shell-hook)"
# preprocess
cd analysis/preprocess
if [ ! -f GSE211692_RAW.tar ]; then
    wget https://www.ncbi.nlm.nih.gov/geo/download/\?acc\=GSE211692\&format\=file -O GSE211692_RAW.tar
else
    echo "GSE211692_RAW.tar already exists, skipping download."
fi
bash run_preprocess.sh -1 GSE211692_RAW.tar work_dir preprocessed_dir mapping_file.txt

# merge all data
python3 tool/get_all_data.py --work_dir ./work_dir/0 --output all_data.csv

# aggreate report
python tool/get_summary.py

# upload to RIVER storage
cd ../../
pwd
mkdir -p $outdir
cp analysis/preprocess/work_dir/all_data.csv $outdir
