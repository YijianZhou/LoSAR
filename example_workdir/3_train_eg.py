""" Training models (main)
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

# i/o paths
cerp_dir = '/home/zhouyj/software/CERP_TED'
shutil.copyfile('config_eg.py', os.path.join(cerp_dir, 'config.py'))
# train params
gpu_idx = 0
num_workers = 10
zarr_path = '/data/bigdata/eg_train-samples.zarr'
ckpt_dir = 'output/eg_ckpt'

# start training
os.system("python {}/train_cnn.py --gpu_idx={} --num_workers={} --zarr_path={} --ckpt_dir={} "\
    .format(cerp_dir, gpu_idx, num_workers, zarr_path, ckpt_dir))
os.system("python {}/train_rnn.py --gpu_idx={} --num_workers={} --zarr_path={} --ckpt_dir={} "\
    .format(cerp_dir, gpu_idx, num_workers, zarr_path, ckpt_dir))
