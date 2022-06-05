## Running Commands

#### NELL-ZS

**DOZSL_RD with Linear Transformation fusion function**
```
python run_K4_linear.py --sim_threshold 0.98 --device 1
```

**DOZSL_RD with Average fusion function**
```
python run_K4_average.py --sim_threshold 0.98 --device 1
```

**DOZSL_AGG with Linear Transformation fusion function**
```
python run_K9_linear.py --sim_threshold 0.98 --device 1
```

**DOZSL_AGG with Average fusion function**
```
python run_K9_average.py --sim_threshold 0.98 --device 1
```


#### Wiki-ZS

**DOZSL_RD with Linear Transformation fusion function**
```
python run_K4_linear.py --DATASET Wiki --input_dim 200 --hidden_layers d100,d --embed_dim 50 --ep_dim 100 --sim_threshold 0.95 --device 2
```
**DOZSL_RD with Average fusion function**
```
python run_K4_average.py --DATASET Wiki --input_dim 200 --hidden_layers d100,d --embed_dim 50 --ep_dim 100 --sim_threshold 0.95 --device 2
```
**DOZSL_AGG with Linear Transformation fusion function**
```
python run_K9_linear.py --DATASET Wiki --input_dim 200 --hidden_layers d100,d --embed_dim 50 --ep_dim 100 --sim_threshold 0.95 --device 2
```
**DOZSL_AGG with Average fusion function**
```
python run_K9_average.py --DATASET Wiki --input_dim 200 --hidden_layers d100,d --embed_dim 50 --ep_dim 100 --sim_threshold 0.95 --device 2
```



