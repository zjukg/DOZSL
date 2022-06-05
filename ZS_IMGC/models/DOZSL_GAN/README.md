## Running Commands

**AwA2**

- Standard ZSL setting:
```
python gan_imgc.py --DATASET AwA2 --ManualSeed 9182 --BatchSize 64 --LR 0.00001 --SynNum 300 --SemEmbed TransE --device 1
```
- Generalized ZSL setting:
```
python gan_imgc.py --DATASET AwA2 --ManualSeed 9182 --BatchSize 64 --LR 0.00001 --GZSL --SynNum 1800 --SemEmbed TransE --device 1
```


**ImNet-A**

- Standard ZSL setting:
```
python gan_imgc.py --DATASET ImNet_A --SemEmbed TransE --device 1
```

- Generalized ZSL setting:
```
python gan_imgc.py --DATASET ImNet_A --GZSL --SemEmbed TransE --device 1
```

**ImNet-O**

- Standard ZSL setting:
```
python gan_imgc.py --DATASET ImNet_O --SemEmbed TransE --device 1
```
- Generalized ZSL setting:
```
python gan_imgc.py --DATASET ImNet_O --GZSL --SemEmbed TransE --device 1
```


