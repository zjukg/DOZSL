## Running Commands

**For AwA (K=2 & K=4)**
```
python run.py --dataset AwA --save_name DisenE_K2_D100_AwA --num_factors 2
python run.py --dataset AwA --save_name DisenE_K4_D100_AwA --num_factors 4

```
**For ImNet_A/O (K=2 & K=4)**
```
python run.py --save_name DisenE_K2_D100_ImNet_A --num_factors 2
python run.py --save_name DisenE_K4_D100_ImNet_A --num_factors 4
python run.py --dataset ImNet_O --save_name DisenE_K2_D100_ImNet_O --num_factors 2
python run.py --dataset ImNet_O --save_name DisenE_K4_D100_ImNet_O --num_factors 4
```
**For NELL-ZS & Wiki-ZS (K=2 & K=4)**
```
python run.py --dataset NELL --save_name DisenE_K2_D200_NELL --num_factors 2 --init_dim 200 --embed_dim 200
python run.py --dataset NELL --save_name DisenE_K4_D200_NELL --num_factors 4 --init_dim 200 --embed_dim 200
python run.py --dataset Wiki --save_name DisenE_K2_D200_Wiki --num_factors 2 --init_dim 200 --embed_dim 200
python run.py --dataset Wiki --save_name DisenE_K4_D200_Wiki --num_factors 4 --init_dim 200 --embed_dim 200
```
