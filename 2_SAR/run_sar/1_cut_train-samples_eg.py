""" Cut training samples in SAC
"""
import os, shutil

# i/o paths
data_dir = '/data/Example_data'
sar_prep_dir = '/home/zhouyj/software/2_SAR/preprocess'
shutil.copyfile('config_eg.py', os.path.join(sar_prep_dir, 'config.py'))
fpha = 'input/eg_pal_hyp.pha'
fpick = 'input/eg_pal.pick'  # all PAL picks
out_root = '/data/bigdata/eg_train-samples_sac'
num_workers = 10

# cut pos & neg
os.system("python {}/cut_positive.py --data_dir={} --fpha={} --out_root={} --num_workers={} "\
    .format(sar_prep_dir, data_dir, fpha, out_root, num_workers))
os.system("python {}/cut_negative.py --data_dir={} --fpha={} --fpick={} --out_root={} --num_workers={} "\
    .format(sar_prep_dir, data_dir, fpha, fpick, out_root, num_workers))