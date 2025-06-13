#!/usr/bin/env bash

#python3 solve2_par.py     --N 50  --t_max 1000 --Q 65 
#python3 maingraf.py       --N 50  --t_max 1000 --Q 65 --delta 0.005
#python3 solve2_par.py     --N 100  --t_max 5000 --Q 65 
#python3 maingraf.py       --N 100  --t_max 5000 --Q 65 
#python3 solve2_par.py     --N 512  --t_max 5000 --Q 65 
#python3 maingraf.py       --N 512  --t_max 5000 --Q 2 --delta 0.05
#python3 maingraf.py       --N 512  --t_max 5000 --Q 2 --delta 0.01
#python3 maingraf.py       --N 512  --t_max 5000 --Q 2 --delta 0.1
#python3 maingraf.py       --N 512  --t_max 5000 --Q 10 --delta 0.05
#python3 maingraf.py       --N 512  --t_max 5000 --Q 10 --delta 0.01
#python3 maingraf.py       --N 512  --t_max 5000 --Q 10 --delta 0.1
#python3 maingraf.py       --N 512  --t_max 5000 --Q 20 --delta 0.05
#python3 maingraf.py       --N 512  --t_max 5000 --Q 20 --delta 0.01
#python3 maingraf.py       --N 512  --t_max 5000 --Q 20 --delta 0.1
#python3 maingraf.py       --N 512  --t_max 5000 --Q 50 --delta 0.05
#python3 maingraf.py       --N 512  --t_max 5000 --Q 50 --delta 0.01
#python3 maingraf.py       --N 512  --t_max 5000 --Q 50 --delta 0.1
#python3 maingraf.py       --N 512  --t_max 5000 --Q 252 --delta 0.1 
#python3 maingraf.py       --N 512  --t_max 5000 --Q 252 --delta 0.005 

python3 123.py     --N 50 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
python3 maingraf.py     --N 50 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
python3 123.py     --N 512 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
python3 maingraf.py     --N 512 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
python3 123.py     --N 512 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05
python3 maingraf.py     --N 50 --gamma 1.0 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05
#python3 main1.py     --N 512 --L 1.0 --gamma 0.3 --V0 0.3     --f 0.8 --kappa 5.2 --alpha 1.457     --t_max 1000 --dt 0.01 --Q 120 --delta 30    --output data4.csv --params params4.json
#python3 main1.py     --N 512 --L 1.0 --gamma 0.3 --V0 0.3     --f 0.8 --kappa 5.2 --alpha 1.457     --t_max 1000 --dt 0.01 --Q 3 --delta 30      --output data5.csv --params params5.json
#python3 main1.py     --N 512 --L 1.0 --gamma 0.3 --V0 0.3     --f 0.8 --kappa 5.2 --alpha 1.457     --t_max 1000 --dt 0.01 --Q 10 --delta 30     --output data5.csv --params params5.json
