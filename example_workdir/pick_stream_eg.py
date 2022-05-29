""" Pick stream data with CERP
"""
import os, shutil

# i/o paths
cerp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_eg.py', os.path.join(cerp_dir, 'config.py'))
fsta = 'input/example_pal_format1.sta'
data_dir = '/data/Example_data'
time_range = '20190704-20190707'
out_root = 'output/eg'
ckpt_dir = 'output/eg_ckpt'
cnn_ckpt = [-1,6000][0]
rnn_ckpt = -1 # -1 for the latest check point
# picking params
gpu_idx = '0'
num_workers = 10

os.system("python {}/run_cerp_stream.py --gpu_idx={} --num_workers={} \
    --data_dir={} --fsta={} --out_root={} --time_range={} \
    --ckpt_dir={} --cnn_ckpt={} --rnn_ckpt={} "\
    .format(cerp_dir, gpu_idx, num_workers, 
    data_dir, fsta, out_root, time_range, 
    ckpt_dir, cnn_ckpt, rnn_ckpt))

