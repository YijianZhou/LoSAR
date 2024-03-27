""" Training SAR model 
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

# i/o paths
sar_dir = '/home/zhouyj/software/LoSAR'
shutil.copyfile('config_eg.py', os.path.join(sar_dir, 'config.py'))
# train params
gpu_idx = 0
num_workers = 10
zarr_path = '/data/bigdata/eg_train-samples.zarr'
ckpt_dir = 'output/eg_ckpt'

# start training
os.system("python {}/train.py --gpu_idx={} --num_workers={} --zarr_path={} --ckpt_dir={} "\
    .format(sar_dir, gpu_idx, num_workers, zarr_path, ckpt_dir))
