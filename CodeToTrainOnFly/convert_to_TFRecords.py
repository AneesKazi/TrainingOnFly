import pydicom as dicom
from pydicom.errors import *
import tensorflow as tf
import numpy as np
import csv
import os
import multiprocessing as mp
import time
import sys
import pandas as pd
import oct_io


def read_csv(csv_path, delimiter=';'):
    """
    method for reading in csv files

    :param csv_path: str
        path to the csv file
    :param delimiter: char
        char that is used for delimiting entries in the csv file
    :return: list
        contents of the csv file
    """

    try:
        # open csv file
        with open(csv_path) as csv_file:

            # read contents and convert to list
            csv_contents = csv.reader(csv_file, delimiter=delimiter)
            csv_list = list(csv_contents)
    except:
        raise IOError("csv file " + csv_path + " could not be opened!")

    return csv_list


def get_paths(imgdir, im_list, identifier):
    selected = [i for i in im_list if i[-1] == identifier]
    paths = [os.path.join(imgdir, 'cimg_db', *i[:4], 'pat.dcm') for i in selected]
    paths = [path for path in paths if os.path.isfile(path)]

    return paths


class CSVToPath(object):

    def __init__(self, root, csv, sep=';', labels_or_indices=None):

        self.frame = pd.read_csv(csv, sep=sep)
        self.root = root
        self.identifiers = labels_or_indices if labels_or_indices is not None else ['HeyexID', 'SeriesID', 'Type2', 'Laterality']
        self.filenames = self.frame.apply(self.build_filename, axis=1)

    def build_filename(self, row):

        filename = self.root
        for idx in self.identifiers:
            filename = os.path.join(filename, str(row[idx]))
        filename = os.path.join(filename, 'pat.dcm')
        return filename


def load_dcm_volume(path):
    """
    method for loading dcm files and extracting the oct volume as a numpy array

    :param path: str
        path to the dicom file
    :return: numpy array
        the loaded oct volume with shape (slice, y, x)
    """
    try:
        # load dicom file and calculate some properties
        dicom_image = dicom.read_file(path)
        n_pix_per_slice = (dicom_image.Rows*dicom_image.Columns)*2
        num_slices = int(len(dicom_image.PixelData)/n_pix_per_slice)
        py, px = dicom_image.Rows, dicom_image.Columns
        # build the oct volume slice by slice from the raw byte data
        oct_volume =[]
        for i in range(num_slices):
            im_raw = np.fromstring(dicom_image.PixelData[i*n_pix_per_slice:(i+1)*n_pix_per_slice], np.int16)
            im_raw = np.reshape(im_raw, (py, px))
            oct_volume.append(im_raw)

        # convert oct volume to numpy array
        oct_volume = np.array(oct_volume, np.int16)

        return oct_volume
    except InvalidDicomError:
        print(path)
        return None


def pad(volume):
    if volume is not None:
        padded = np.pad(volume,[(0,0),(8,8),(0,0)],'symmetric')
        return padded


def combine(path):
    # print('volume loaded: {}'.format(path.split('/')[-5:]))
    return pad(load_dcm_volume(path))


def load(path):
    print('loading: ' + path)
    try:
        oct = oct_io.OCTVolume(path, load=True, pad={}, normalize=True)
    except InvalidDicomError:
        print('invalid file:'+path)
        return None
    return oct.image_data


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def construct_shard_names(basename, n_shards):
    n_digits = len(str(n_shards))+1
    names = []
    for shard in range(1, n_shards+1):
        names.append('{}_{}_of_{}.tfrecords'.format(basename, str(shard).zfill(n_digits), str(n_shards).zfill(n_digits)))
    return names


def crop_and_display(quality):
    import matplotlib.pyplot as plt
    # settings
    imgdir = '/media/network/ImageRepository/'
    csv_file = os.path.join('/media/network/ImageRepository', 'imanakov', 'Dataset-low-high.csv')

    # initialize workers and paths
    csv_file = read_csv(csv_file)
    lq_paths = get_paths(imgdir, csv_file, quality)

    for filename in lq_paths:
        # load and preprocess volumes
        lq = combine(filename)

        for slice in lq:

            slice[slice>0] = 0
            slice -= slice.min()
            slice = slice / slice.max()
            # plt.hist(slice.ravel(), 100)
            min, max = 0.5, 0.999
            slice[slice<min] = 0
            slice[slice>max] = 0
            plt.imshow(slice, vmin=min, vmax=max, cmap='gray')
            plt.show()


def prep_slice(slice, min, max):
    slice[slice > 0] = 0
    slice -= slice.min()
    slice = slice / slice.max()
    # # plt.hist(slice.ravel(), 100)
    # slice[slice < min] = 0
    # slice[slice > max] = 0
    return slice


def crop_and_compare():
    import matplotlib.pyplot as plt
    # settings
    imgdir = '/media/network/ImageRepository/'
    csv_file = os.path.join('/media/network/ImageRepository', 'imanakov', 'Dataset-low-high.csv')

    # initialize workers and paths
    csv_file = read_csv(csv_file)
    lq_paths = get_paths(imgdir, csv_file, 'low')
    hq_paths = get_paths(imgdir, csv_file, 'high')

    for lq_file, hq_file in zip(lq_paths, hq_paths):
        # load and preprocess volumes
        lq = combine(lq_file)
        hq = combine(hq_file)

        for lqs, hqs in zip(lq, hq):

            min, max = 0.5, 0.99
            slice = np.c_[prep_slice(lqs, min, max), prep_slice(hqs, min, max)]
            plt.figure(figsize=(32,16), dpi=32)
            plt.imshow(slice, cmap='gray')#, vmin=min, vmax=max, cmap='gray')
            plt.axis('off')
            plt.show()
            plt.close()


if __name__ == '__main__':
    #
    # crop_and_compare()
    #
    # exit(0)
    # settings
    num_files_out = 32
    quality = sys.argv[3]
    imgdir = sys.argv[1]
    destdir = sys.argv[2]
    tf_filename = os.path.join(destdir, 'HQ_LQ_OCTs', 'OCT_Quality_TFRecords2')
    if not os.path.exists(tf_filename):
        os.makedirs(tf_filename)
    tf_filename = os.path.join(tf_filename, '{}'.format(quality))
    tf_filenames = construct_shard_names(tf_filename, num_files_out)
    csv_file = os.path.join(imgdir, 'imanakov', 'Dataset-low-high.csv')
    cpus = mp.cpu_count()
    start = time.time()

    # initialize workers and paths
    csv_file = read_csv(csv_file)
    pool = mp.Pool(cpus)
    paths = get_paths(imgdir, csv_file, quality)
    path_shards = []
    files_per_shard = len(paths) // num_files_out
    for i in range(num_files_out):
        if i != num_files_out-1:
            path_shards.append(paths[i*files_per_shard:(i+1)*files_per_shard])
        else:
            path_shards.append(paths[i*files_per_shard:])


    # create tfrecords shards
    for ind, filename in enumerate(tf_filenames):
        start = time.time()
        paths = path_shards[ind]
        print('working on:')
        print(paths)
        print()
        # load and preprocess volumes
        volumes = pool.map(load, paths)
        volumes = [v for v in volumes if v is not None]
        volumes = np.concatenate(volumes)

        # convert to raw bytes
        volumes = [i.tostring() for i in volumes]
        # volume = np.array_split(volume, num_files_out)

        end = time.time()
        print('done loading an processing.\tduration: {}s'.format(int(end - start)))

        duration = 0

        with tf.python_io.TFRecordWriter(filename) as tf_writer:
            for im in volumes:
                # convert to tf format
                data = {'image': wrap_bytes(im)}
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                tf_writer.write(serialized)

        end = time.time()
        duration += end-start
        print('finished batch {} out of {}.\tduration: {}s'.format(ind, num_files_out, round(end - start, 3)))

    print('finished conversion.\tduration: {}s'.format(duration))




