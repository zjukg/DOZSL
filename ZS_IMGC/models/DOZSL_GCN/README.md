## Running Commands

#### Taking ImNet-A as an example

**DOZSL_RD with Linear Transformation fusion function**
- Standard ZSL
```
python run_K2_linear.py --sim_threshold 0.98 --DATASET ImageNet/ImNet_A --device 1
```
- Generalized ZSL
```
python run_K2_linear.py --sim_threshold 0.98 --DATASET ImageNet/ImNet_A --gzsl --device 1
```

**DOZSL_RD with Average fusion function**
```
python run_K2_average.py --sim_threshold 0.98 --DATASET ImageNet/ImNet_A --device 1
```

**DOZSL_AGG with Linear Transformation fusion function**
```
python run_K5_linear.py --sim_threshold 0.98 --DATASET ImageNet/ImNet_A --device 1
```
**DOZSL_AGG with Average fusion function**
```
python run_K5_average.py --sim_threshold 0.98 --DATASET ImageNet/ImNet_A --device 1
```

