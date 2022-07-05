# CERP_Pytorch
**C**NN-**E**vent detector & **R**NN-**P**hase picker (CERP), implemented in Pytorch <br>

## Usage  <br>
### 1. Train CERP model <br>
**1.1 run [PAL](https://github.com/YijianZhou/PAL) to generate local training samples**  <br>
**1.2 cut event windows & generate [Zarr](https://zarr.readthedocs.io/en/stable/) database**  <br>
```bash
python cut_train-samples.py
python sac2zarr.py
```  
**1.3 train CERP model**  <br>
```bash
python train.py
```
### 2. Apply CERP on continuous data & associate picks <br>
**2.1 run CERP**
```bash
python pick_stream.py
```  
**2.2 associate picks with PAL associator**
```bash
python parallel_assoc.py
```  
<br>

### Reference <br>
- **Zhou, Y.**, H. Yue, Q. Kong, & S. Zhou (2019). Hybrid Event Detection and Phase‐Picking Algorithm Using Convolutional and Recurrent Neural Networks. *Seismological Research Letters*; 90 (3): 1079–1087. doi: [10.1785/0220180319](https://doi.org/10.1785/0220180319)
- **Zhou, Y.**, H. Yue, L. Fang, S. Zhou, L. Zhao, & A. Ghosh (2021). An Earthquake Detection and Location Architecture for Continuous Seismograms: Phase Picking, Association, Location, and Matched Filter (PALM). *Seismological Research Letters*; 93(1): 413–425. doi: [10.1785/0220210111](https://doi.org/10.1785/0220210111)  
