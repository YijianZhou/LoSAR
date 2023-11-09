""" Training RSeL model 
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

# i/o paths
rsel_dir = '/home/zhouyj/software/RSeL_TED'
shutil.copyfile('config_eg.py', os.path.join(rsel_dir, 'config.py'))
# train params
gpu_idx = 0
num_workers = 10
hdf5_path = '/data/bigdata/eg_train-samples.h5'
ckpt_dir = 'output/eg_ckpt'

# start training
os.system("python {}/train.py --gpu_idx={} --num_workers={} --hdf5_path={} --ckpt_dir={} "\
    .format(rsel_dir, gpu_idx, num_workers, hdf5_path, ckpt_dir))
