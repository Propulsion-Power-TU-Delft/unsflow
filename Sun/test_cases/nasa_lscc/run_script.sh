cd 10_40_30_ur || exit
echo SIMULATION UR
python main_compressor.py
cd ../10_40_30_ut || exit
echo SIMULATION UT
python main_compressor.py
cd ../10_40_30_uz || exit
echo SIMULATION UT
python main_compressor.py
cd ..
