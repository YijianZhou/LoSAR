""" Pick stream data 
"""
import os, shutil

# i/o paths
sar_dir = '/home/zhouyj/software/SAR_TED'
shutil.copyfile('config_eg.py', os.path.join(sar_dir, 'config.py'))
fsta = 'input/example_pal_format1.sta'
data_dir = '/data/Example_data'
time_range = '20190704-20190707'
out_root = 'output/eg'
# picking params
gpu_idx = 0
num_workers = 10
ckpt_dir = 'output/eg_ckpt'
ckpt_idx = -1  # -1 for the latest check point

os.system("python {}/run_picker.py --gpu_idx={} --num_workers={} \
    --data_dir={} --fsta={} --out_root={} --time_range={} --ckpt_dir={} --ckpt_idx={} "\
    .format(sar_dir, gpu_idx, num_workers, 
    data_dir, fsta, out_root, time_range, ckpt_dir, ckpt_idx))

