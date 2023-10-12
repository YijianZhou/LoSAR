""" Cut training samples in SAC
"""
import os, shutil

# i/o paths
data_dir = '/data/Example_data'
rsel_prep_dir = '/home/zhouyj/software/RSeL_TED/preprocess'
shutil.copyfile('config_eg.py', os.path.join(rsel_prep_dir, 'config.py'))
fpha = 'input/eg_pal_hyp.pha'
fpick = 'input/eg_pal.pick' 
out_root = '/data/bigdata/eg_train-samples_sac'
# cut params
cut_method = 'long' # 'intense' or 'long'
num_workers = 10

# cut pos & neg
os.system("python {}/cut_pos_{}.py --data_dir={} --fpha={} --out_root={} --num_workers={} "\
    .format(rsel_prep_dir, cut_method, data_dir, fpha, out_root, num_workers))
os.system("python {}/cut_neg_{}.py --data_dir={} --fpha={} --fpick={} --out_root={} --num_workers={} "\
    .format(rsel_prep_dir, cut_method, data_dir, fpha, fpick, out_root, num_workers))