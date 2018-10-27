import sys
sys.path.append('/home/kazuki/Documents/Promotion/Project_Helpers/utils/')
import models
import os
import oct_io
import pandas as pd
import tensorflow as tf
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


def build_filenames(row, root, columns):

    filename = root

    for column in columns:
        filename = os.path.join(filename, str(row[column]))
    else:
        filename = os.path.join(filename, 'pat.dcm')

    return filename

def export_image(image, filename):

    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray', vmin=0.7, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# parameters
checkpoint = 'cGAN.ckpt-1400562'
model_dir = '/media/network/DL_PC/ilja/cycoct-alldata-threshold/gen_norm-in_act-selu_scale-3_res-6x3_f-16/dis_norm-in_act-selu_f-1/10-cyc_1-dis/'
export_dir = '/media/network/aug-learning/cGAN_subjective-evaluation/'
csv_file = '/media/network/aug-learning/users/imanakov/Validation-low-high.csv'
file_root = '/media/network/ImageRepository/cimg_db'
path_columns = ['PID', 'SID', 'ExamType1', 'Position']

# load model
checkpoint = os.path.join(model_dir, checkpoint)
inputs = {'hq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'HQ-Input'),
          'lq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'LQ-Input')}
gen_arch = model_dir + 'generator_architecture.pkl'
dis_arch = model_dir + 'discriminator_architecture.pkl'
model = models.CyclicGAN.from_trained_model(inputs, gen_arch, dis_arch, checkpoint)

# read list of OCT files
file_frame = pd.read_csv(csv_file, sep=';')
file_frame['Filename'] = file_frame.apply(partial(build_filenames, root=file_root, columns=path_columns), axis=1)
file_frame.set_index(['PID', 'Position', 'Quality'], inplace=True)

# randomize indices for each slice (will serve as folder names)
num_files = len(file_frame) / 2
indices_lq = np.arange(num_files * 49 + 1, dtype=np.int16)[0:]
np.random.shuffle(indices_lq)
indices_lq = list(indices_lq)
indices_hq = np.arange(num_files * 49 + 1, dtype=np.int16)[0:]
np.random.shuffle(indices_hq)
indices_hq = list(indices_hq)

# set directories for export
export_dir_lq = os.path.join(export_dir, 'LQ-Comparison')
export_dir_hq = os.path.join(export_dir, 'HQ-Comparison')

# initialize record of exported slices
export_record = []

for (pid, pos), frame in file_frame.groupby(level=[0, 1]):

    # load OCTs
    lq_volume = oct_io.OCTVolume(frame.loc[pid, pos, 'low']['Filename'], load=True, pad={}, normalize={'min': 'mean-0.5'}).image_data
    hq_volume = oct_io.OCTVolume(frame.loc[pid, pos, 'high']['Filename'], load=True, pad={}, normalize={'min': 'mean-0.5'}).image_data

    for lq, hq in zip(lq_volume, hq_volume):

        # enhance image
        enhanced = model.generate(lq[np.newaxis, ..., np.newaxis], 'hq')[0, ..., 0]

        # export LQ + Enhanced
        index = str(indices_lq.pop()).zfill(6)
        path = os.path.join(export_dir_lq, index)
        if not os.path.isdir(path): os.makedirs(path)
        entry = np.random.randint(1, 3)
        export_image(enhanced, os.path.join(path, str(entry) + '.png'))
        entry = 3 - entry
        export_image(lq, os.path.join(path, str(entry) + '.png'))
        export_record.append({'Test': 'LQ-Comparison', 'enhanced': 3 - entry, 'original': entry})

        # export Enhanced + HQ
        index = str(indices_hq.pop()).zfill(6)
        path = os.path.join(export_dir_hq, index)
        if not os.path.isdir(path): os.makedirs(path)
        entry = np.random.randint(1, 3)
        export_image(enhanced, os.path.join(path, str(entry) + '.png'))
        entry = 3 - entry
        export_image(hq, os.path.join(path, str(entry) + '.png'))
        export_record.append({'Test': 'HQ-Comparison', 'enhanced': 3 - entry, 'original': entry})

    export_record = pd.DataFrame(export_record)
    export_record.to_csv(os.path.join(export_dir, 'export-record.csv'))

    exit(0)


