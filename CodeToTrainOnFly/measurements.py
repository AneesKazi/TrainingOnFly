from skimage import filters, measure, restoration
import numpy as np
import os
import pandas as pd
from functools import partial
import oct_io as io
import matplotlib.pyplot as plt
from registrator import Registrator
from investigate import Denoiser
from pathos.helpers import mp
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.FATAL)
plt.ioff()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Datapoint(object):

    def __init__(self, template, index=None, image=None, reference=None, method=None, background=None, rois=None):

        self.data = template.copy()
        self.slice = index
        self.method = method
        self.image = image.copy()
        self.reference = reference
        self.rois = rois
        self.background = background
        self.contains_measurement = False

    def extract_information(self):

        # test if some data is missing
        assert None not in self.data.values(), f'some data is still missing: {self.data}!'
        assert self.contains_measurement, f'no measurement was performed on this datapoint!'

        return self.data

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        assert image.shape == (512,512), f'image dimesions are {image.shape}, but should be 512x512'
        self._image = image

    @property
    def slice(self):
        return self.data['Slice']

    @slice.setter
    def slice(self, index):
        assert isinstance(index, int), f'index must be integer but is {type(index)}'
        self.data['Slice'] = index

    @property
    def method(self):
        return self.data['Method']

    @method.setter
    def method(self, method):
        assert isinstance(method, str), f'method name must be string but is {type(method)}'
        self.data['Method'] = method

    def add_measurement(self, name, value):
        self.data[name] = value
        self.contains_measurement = True

    def copy(self):

        new_datapoint = Datapoint(template=self.data,
                                  image=self.image,
                                  reference=self.reference,
                                  rois=self.rois,
                                  background=self.background,
                                  index=self.slice,
                                  method=self.method)

        return new_datapoint


def show(image, **kwargs):
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    plt.imshow(image, **kwargs)
    plt.axis('off')
    plt.show()


def hist(image):
    fig = plt.figure(figsize=(8, 8))
    plt.hist(image.ravel(), 100, range=(0.1, 1))
    plt.show()


def bilateral(image):
    sigma = restoration.estimate_sigma(image[image > 0])
    denoised = restoration.denoise_bilateral(image, sigma_color=sigma, multichannel=False)
    return denoised


def wavelet(image):
    sigma = restoration.estimate_sigma(image[image > 0])
    denoised = restoration.denoise_wavelet(image, multichannel=False, method='BayesShrink', sigma=sigma, mode='soft')
    return denoised


def nl_means(image):
    sigma = restoration.estimate_sigma(image[image > 0])
    denoised = restoration.denoise_nl_means(image, h=sigma/2, patch_distance=11, multichannel=False)
    return denoised


def median(image, filter_size=5):
    filter = np.ones((filter_size, filter_size))
    image = filters.median(image, selem=filter)

    return image


class CycGAN(object):

    def __init__(self, checkpoint, model_path):

        self.denoiser = Denoiser(checkpoint, model_path)

    def __call__(self, image):

        return self.denoiser.denoise(image)


def psnr(datapoint):
    return measure.compare_psnr(datapoint.reference, datapoint.image)


def ssim(datapoint):
    return measure.compare_ssim(datapoint.image, datapoint.reference)


def cnr(datapoint):

    rois, background = [datapoint.image[roi > 0] for roi in datapoint.rois], datapoint.image[datapoint.background > 0]
    background_mean = background.mean()
    background_std = background.std()
    cnrs = []
    for roi in rois:
        cnrs.append(np.abs(roi.mean() - background_mean) / np.sqrt(0.5*(roi.std()**2 + background_std)**2))
    cnrs = np.array(cnrs)

    return cnrs.mean()


def msr(datapoint):

    rois = [datapoint.image[roi > 0] for roi in datapoint.rois]
    msrs = []
    for roi in rois:
        mean = roi.mean()
        std = roi.std()
        msrs.append(mean/std)
    msrs = np.array(msrs)

    return msrs.mean()


# define constants
PATH_COLUMNS = ['PID', 'SID', 'ExamType1', 'Position']
INDEX = ['PID', 'Position', 'Quality']
IMAGE_REPOSITORY = '/media/network/ImageRepository/cimg_db'
AUG_LEARNING = '/media/network/aug-learning'
DL_PC = '/media/network/DL_PC/ilja'
CSV = './Validation-low-high_testing.csv'
# CSV = '/media/network/aug-learning/users/imanakov/Validation-low-high.csv'
MODEL = os.path.join(DL_PC, 'cycoct-alldata/gen_norm-in_act-selu_scale-3_res-6x3_f-16/dis_norm-in_act-selu_f-1/10-cyc_1-dis/')
CHECKPOINT = 'cGAN.ckpt-101490'
# METHODS = {'median': median,
#            'ours': CycGAN(CHECKPOINT, MODEL),
#            'wavelet': wavelet,
#            'bilateral': bilateral,
#            'nl_means': nl_means}
METHODS = {'median': median,
           'wavelet': wavelet,
           'bilateral': bilateral,
           'nl_means': nl_means}
MEASUREMENTS = {'PSNR': psnr,
                'CNR': cnr,
                'MSR': msr,
                'SSIM': ssim}
# SAVEFILE = os.path.join(DL_PC, 'cycoct-results', 'measurements.csv')
SAVEFILE = './measurements.csv'


def build_filenames(row, root, columns):

    filename = root

    for column in columns:
        filename = os.path.join(filename, str(row[column]))
    else:
        filename = os.path.join(filename, 'pat.dcm')

    return filename


class Analysis(object):

    def __init__(self, image_repository, path_columns, savefile,
                 registrator=None, n_processes=mp.cpu_count(), debug=False):

        self.debug = debug
        print('initializing analysis...',
              '\timage repository:\t{}'.format(image_repository),
              '\tpath columns:\t{}'.format(path_columns),
              '\tsavefile:\t{}'.format(savefile),
              '\tprocesses:\t{}'.format(n_processes),
              '\tmeasurements:\{}'.format(list(MEASUREMENTS.keys())),
              '\tdenoising methods:\{}'.format(list(METHODS.keys())), sep='\n')

        self.methods = METHODS.copy()
        self.measurements = MEASUREMENTS.copy()

        self.savefile = savefile
        self.image_repository = image_repository
        self.path_columns = path_columns
        self.pool = mp.Pool(n_processes)
        self.registrator = registrator if isinstance(registrator, Registrator) else Registrator(verbose=debug, graphic=debug)
        self.denoising = None

        # make save dir if it does not exist
        save_path = os.path.dirname(self.savefile)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        print('done!\n')

    def get_rois(self, image):

        masked = self.registrator.segment(image, offset=1)

        masked = np.pad(masked, ((6, 6),), 'constant', constant_values=0)
        contours = self.registrator.get_contours(masked, offset=0.4, min_length=0.1)
        masks = [self.registrator.get_mask(contour, masked.shape) for contour in contours]
        if self.debug:
            for i, mask in enumerate(masks):
                plt.imshow(masked * mask, cmap='gray')
                plt.axis('off')
                plt.title('Mask {}'.format(i))
                plt.show()

        return masks

    def get_background(self, image):

        inverted_mask = np.ones_like(image)
        inverted_mask[self.registrator.segment(image, offset=-1) > 0] = 0
        if self.debug:
            plt.imshow(image * inverted_mask, cmap='gray')
            plt.axis('off')
            plt.title('Background')
            plt.show()

        return inverted_mask

    def __call__(self, csv):

        print('beginning analysis...',
              '\tcsv:\t{}'.format(csv), sep='\n')

        # set up input dataframe with data and filenames
        data = pd.DataFrame.from_csv(csv, sep=',', index_col=None)
        data['Filename'] = data.apply(partial(build_filenames, root=self.image_repository, columns=self.path_columns), axis=1)
        data.set_index(['PID', 'Position', 'Quality'], inplace=True)

        total_slices = data.loc[:, 'ImgCount'].sum() / 4
        analized_slices = 0

        print('\tnumber of datapoins:\t{}'.format(len(data.index)))

        # initialize result list for accumulating measurements
        results = []

        # cycle over volumes in test set
        for (pid, pos), frame in data.groupby(level=[0, 1]):

            # prepare template for results
            result_template = {'PID': pid,
                        'SID': frame.loc[pid, pos, 'low']['SID'],
                        'Slice': None,
                        'Position': pos,
                        'ExamType1': frame.loc[pid, pos, 'low']['ExamType1'],
                        'Filename': frame.loc[pid, pos, 'low']['Filename'],
                        'Method': None}

            # load OCTs
            lq = io.OCTVolume(frame.loc[pid, pos, 'low']['Filename'], load=True, pad={}, normalize=True).image_data
            hq = io.OCTVolume(frame.loc[pid, pos, 'high']['Filename'], load=True, pad={}, normalize=True).image_data

            # register volumes
            lq = self.registrator.register_volume(hq, lq, offsets=0.5)

            # cycle over b-scans in volume
            for index, (image, reference) in enumerate(zip(lq, hq)):

                # skip failed registrations
                if image is None:
                    analized_slices += 1
                    continue

                rois = self.get_rois(reference)
                background = self.get_background(reference)

                # initialize dict for the denoised images
                datapoints = [Datapoint(template=result_template, index=index, reference=reference,
                                        rois=rois, background=background, method='original', image=image)]

                # generate all denoised images
                # self.denoising = partial(self.apply_denoising, datapoint=datapoints[0])
                # datapoints += self.pool.map(self.denoising, self.methods.keys())
                datapoints += [self.apply_denoising(datapoints[0], method) for method in self.methods.keys()]

                # perform measurements on all denoised images
                # results += self.pool.map(self.apply_measurements, datapoints)
                results += [self.apply_measurements(datapoint) for datapoint in datapoints]

                analized_slices += 1
                print('\r\tprogress: \t{}%'.format(round(100*analized_slices/total_slices, 2)))

        # convert results to a dataframe and save
        results_frame = pd.DataFrame(data=results)
        results_frame.to_csv(SAVEFILE)

        return results_frame

    def apply_measurements(self, datapoint):

        for measurement_name, measurement in self.measurements.items():

            result = measurement(datapoint)
            datapoint.add_measurement(measurement_name, result)

        result = datapoint.extract_information()
        return result

    def apply_denoising(self, datapoint, method_name):

        method = self.methods[method_name]
        datapoint = datapoint.copy()
        datapoint.image = method(datapoint.image)
        datapoint.method = method_name

        return datapoint


if __name__ == '__main__':

    analysis = Analysis(IMAGE_REPOSITORY, PATH_COLUMNS, SAVEFILE, debug=True)
    results = analysis(CSV)
