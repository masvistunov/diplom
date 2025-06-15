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


#если считаем что -1/2
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 20 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 20 --delta 0.05
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 20 --delta 0.1
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 20 --delta 0.1
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 30 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 30 --delta 0.05
#python3 123.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 60 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 60 --delta 0.05



#если считаем что -3/2
python3 123.py     --N 100 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 3
python3 maingraf.py     --N 100 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 3

python3 123.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 3
python3 maingraf.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 3
python3 123.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05 --stepen 3
python3 maingraf.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05 --stepen 3
python3 123.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 60 --delta 0.05 --stepen 3
python3 maingraf.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 60 --delta 0.05 --stepen 3
python3 123.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 100 --delta 0.05 --stepen 3
python3 maingraf.py     --N 512 --gamma 3.0 --v0 0.05    --t_max 5000 --dt 0.01 --Q 100 --delta 0.05 --stepen 3

# -1/2
#python3 123.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 1
#python3 maingraf.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05 --stepen 1
#python3 123.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05 --stepen 1
#python3 maingraf.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 5 --delta 0.05 --stepen 1
#python3 123.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05 --stepen 1
#python3 maingraf.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05 --stepen 1
#python3 123.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 25 --delta 0.05 --stepen 1
#python3 maingraf.py     --N 512 --gamma 2.5 --v0 0.12    --t_max 5000 --dt 0.01 --Q 25 --delta 0.05 --stepen 1


#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 2 --delta 0.05
#python3 123.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 3 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 3 --delta 0.05
#python3 123.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.1
#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.1
#python3 123.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 15 --delta 0.1
#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 15 --delta 0.1
#python3 123.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 10 --delta 0.05
#python3 123.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 30 --delta 0.05
#python3 maingraf.py     --N 512 --gamma 1.6 --v0 0.12    --t_max 5000 --dt 0.01 --Q 30 --delta 0.05
