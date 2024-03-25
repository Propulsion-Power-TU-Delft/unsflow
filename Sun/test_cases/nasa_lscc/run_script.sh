cd 20_40_40_20_ur || exit
echo SIMULATION UR
echo Progress being printed on log.txt
python main_compressor.py > log.txt
cd ../20_40_40_20_ut || exit
echo SIMULATION UT
echo Progress being printed on log.txt
python main_compressor.py > log.txt
cd ../20_40_40_20_uz || exit
echo SIMULATION UZ
echo Progress being printed on log.txt
python main_compressor.py > log.txt
cd ..
