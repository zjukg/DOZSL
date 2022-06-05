## Running Commands

**For AwA & ImNet_A/O**
```
python run.py --dataset AwA --save_name RGAT_D100_AwA --init_dim 100 --gcn_dim 100 --embed_dim 100
python run.py --dataset ImNet_A --save_name RGAT_D100_ImNet_A --init_dim 100 --gcn_dim 100 --embed_dim 100
python run.py --dataset ImNet_O --save_name RGAT_D100_ImNet_O --init_dim 100 --gcn_dim 100 --embed_dim 100
```
**For NELL-ZS & Wiki-ZS**
```
python run.py --dataset NELL --save_name RGAT_D200_NELL --init_dim 200 --gcn_dim 200 --embed_dim 200
python run.py --dataset Wiki --save_name RGAT_D200_Wiki --init_dim 200 --gcn_dim 200 --embed_dim 200

```