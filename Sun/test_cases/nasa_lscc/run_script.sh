cd 20_40_30_ur || exit
#echo SIMULATION UR
#echo Progress being printed on log.txt
#python main_compressor.py > log.txt
cd ../20_40_30_ut || exit
echo SIMULATION UT
echo Progress being printed on log.txt
python main_compressor.py > log.txt
cd ../20_40_30_uz || exit
echo SIMULATION UT
echo Progress being printed on log.txt
python main_compressor.py > log.txt
cd ..
