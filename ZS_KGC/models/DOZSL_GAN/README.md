## Running Commands

**for NELL-ZS**
```
python gan_kgc.py --device 1
```

**for Wiki-ZS**
```
python gan_kgc.py --dataset Wiki --embed_dim 50 --ep_dim 100 --fc1_dim 200 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8 --SemEmbed TransE --device 1
```



