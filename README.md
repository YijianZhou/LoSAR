# CERP_Pytorch
**C**NN-**E**vent detector & **R**NN-**P**hase picker (CERP), implemented in Pytorch 

##Usage <br>
**1. Training CERP model** <br>
    1.1 run [PAL](https://github.com/YijianZhou/PAL) to generate local training samples <br>
    1.2 cut event windows & generate Zarr database <br>
```bash
python cut_train-samples.py
python sac2zarr.py
```

    1.3 train CERP model
```bash
python train.py
```

**2. Apply CERP on continuous data & associate picks** <br>
    2.1 run CERP
```bash
python pick_stream.py
```

    2.2 associate picks with PAL associator
```bash
python parallel_assoc.py
```
<br>
### Reference <br>
**Yijian Zhou**, Han Yue, Qingkai Kong, Shiyong Zhou; Hybrid Event Detection and Phase‐Picking Algorithm Using Convolutional and Recurrent Neural Networks. Seismological Research Letters 2019;; 90 (3): 1079–1087. doi: https://doi.org/10.1785/0220180319
