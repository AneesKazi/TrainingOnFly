import oct_io as io
import models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy


class Denoiser(object):

    def __init__(self, checkpoint, basename):
        # load model
        self.basename = basename
        self.checkpoint = basename + checkpoint
        self.garch = basename + 'generator_architecture.pkl'
        self.darch = basename + 'discriminator_architecture.pkl'
        self.inputs = {'hq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'HQ-Input'),
                       'lq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'LQ-Input')}
        self.model = models.CyclicGAN.from_trained_model(self.inputs, self.garch, self.darch, self.checkpoint)

    def denoise(self, image):

        return self.model.generate(image[np.newaxis, ..., np.newaxis], 'HQ')[0, ..., 0]


class Investigator(object):

    def __init__(self, checkpoint, basename='', sample_files='', cmap='gray', figsize=(8, 4)):
        # load model
        self.basename = '/media/network/DL_PC/ilja/cycoct-alldata/gen_norm-in_act-selu_scale-3_res-6x3_f-16/dis_norm-id_act-selu_f-1/1-cyc_1-dis/' if not basename else basename
        self.checkpoint = basename + checkpoint
        self.garch = basename + 'generator_architecture.pkl'
        self.darch = basename + 'discriminator_architecture.pkl'
        # garch = cg.CyclicGAN.generate_gen_arch(3, 6, 3, tf.nn.selu, filters=16, skip_conn=True)
        # darch = cg.CyclicGAN.generate_dis_arch(512, tf.nn.selu, norm_func=cg.identity)
        self.logdir = '/home/kazuki/testing'
        self.inputs = {'hq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'HQ-Input'),
                       'lq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'LQ-Input')}
        self.model = models.CyclicGAN.from_trained_model(self.inputs, self.garch, self.darch, self.checkpoint)
        self.graph = self.model.graph
        # self.graph = tf.get_default_graph()
        # self.model = cg.CyclicGAN(garch, darch, self.inputs, None, logdir, graph=self.graph, inference_mode=True)
        # self.model.load(self.checkpoint)
        sample_files = sample_files if sample_files else ['/media/network/ImageRepository/cimg_db/24573/564875/6/R/pat.dcm', '/media/network/ImageRepository/cimg_db/24573/564879/6/R/pat.dcm']
        # self.samples = samples if samples else ['../DeepOCTPrior/test_octs/lo.dcm', '../DeepOCTPrior/test_octs/hi.dcm']

        # load in sample dicoms
        self.samples = {}
        self.load_sample(*sample_files)

        # set pyplot parameters
        self.figsize = figsize
        self.cmap = cmap

    def change_checkpoint(self, checkpoint):
        self.checkpoint = self.basename + checkpoint
        self.model.load(self.checkpoint)

    def load_sample(self, lq, hq):

        self.sample_files = [lq, hq]
        lq_oct = io.OCTVolume(lq, load=True, pad={}, normalize={'min': 'mean-0.5'})
        hq_oct = io.OCTVolume(hq, load=True, pad={}, normalize={'min': 'mean-0.5'})

        hq = np.reshape(hq_oct.image_data, (49, 512, 512, 1))
        lq = np.reshape(lq_oct.image_data, (49, 512, 512, 1))

        self.samples = {'hq': hq, 'lq': lq}

    def generate(self, source_index, target, source='', graphical=False):

        if not source:
            source = target
        sample = self.samples[source][source_index:source_index + 1]
        result = self.model.generate(sample, target)
        output = np.c_[sample[0,...,0], result[0,...,0]]
        if graphical:
            plt.figure(figsize=self.figsize)
            plt.imshow(output, cmap=self.cmap)
            plt.show()

        return result

    def show_activations(self, source_index, model_part, target, source=''):

        # helper for domain input
        complement = 'hq' if target == 'lq' else 'lq'
        if not source:
            source = complement

        # get relevant layers
        if model_part == 'gen':
            if target == 'hq':
                layers = self.model.hq.gen.layers
            elif target == 'lq':
                layers = self.model.lq.gen.layers
        elif model_part == 'dis':
            if target == 'hq':
                layers = self.model.hq.dis.layers
            elif target == 'lq':
                layers = self.model.lq.dis.layers
        else:
            raise ValueError('unknown model part or target: {}, {}'.format(model_part, target))

        sample = self.samples[source][source_index:source_index + 1]
        activations = self.model.sess.run(layers, {layers[0]: sample})
        [mosaic(act[0,...], cmap=self.cmap, figsize=self.figsize) for act in activations]

        return activations

    def modify(self, layer_index, channel_index, model_part, target, source='', source_index=0):

        # get relevant layers
        if model_part == 'gen':
            if target == 'hq':
                layers = self.model.hq.gen.layers
            elif target == 'lq':
                layers = self.model.lq.gen.layers
        elif model_part == 'dis':
            if target == 'hq':
                layers = self.model.hq.dis.layers
            elif target == 'lq':
                layers = self.model.lq.dis.layers
        else:
            raise ValueError('unknown model part or target: {}, {}'.format(model_part, target))

        # helper for domain input
        complement = 'hq' if target == 'lq' else 'lq'

        # setup sample
        if not source:
            source = complement
        sample = self.samples[source][source_index:source_index + 1]

        # get tensor for the specified layer
        modified = self.model.sess.run(layers[layer_index], {layers[0]: sample})

        # drop channel
        modified[..., channel_index] = np.zeros((modified.shape[:-1]))

        # generate from modified tensor
        result = self.model.sess.run(layers[-1], {layers[layer_index]: modified})

        # show result
        output = np.c_[sample[0, ..., 0], result[0, ..., 0]]
        plt.figure(figsize=self.figsize)
        plt.imshow(output, cmap=self.cmap)
        plt.show()


def setup():

    # load model
    checkpoint = '/media/network/DL_PC/ilja/cycoct-skip_processed/gen_norm-in_act-selu_scale-3_res-6x3_f-16/dis_norm-id_act-selu_f-1/10-cyc_1-dis/cGAN.ckpt-333200'
    garch = models.CyclicGAN.generate_gen_arch(3, 6, 3, tf.nn.selu, filters=16, skip_conn=True)
    darch = models.CyclicGAN.generate_dis_arch(512, tf.nn.selu, norm_func=models.identity)
    logdir = '/home/kazuki/testing'
    inputs = {'hq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'HQ-Input'),
              'lq': tf.placeholder(tf.float32, (1, 512, 512, 1), 'LQ-Input')}
    graph = tf.get_default_graph()
    cgan = models.CyclicGAN(garch, darch, inputs, None, logdir, graph=graph, inference_mode=True)

    # load in sample dicoms
    # lq_oct = io.OCTVolume('../DeepOCTPrior/test_octs/lo.dcm')
    # hq_oct = io.OCTVolume('../DeepOCTPrior/test_octs/hi.dcm')
    lq_oct = io.OCTVolume('/media/network/ImageRepository/cimg_db/24573/564875/6/R/pat.dcm')
    hq_oct = io.OCTVolume('/media/network/ImageRepository/cimg_db/24573/564879/6/R/pat.dcm')
    lq_oct.load()
    hq_oct.load()
    hq_oct.pad()
    lq_oct.pad()

    return hq_oct, lq_oct, cgan, checkpoint, inputs


def convert_to_uint8(nparray):

    nparray += np.abs(nparray.min())
    nparray /= nparray.max()
    nparray *= 255
    nparray = nparray.astype(np.uint8)

    return nparray


def mosaic(volume, **plot_args):
    import math
    n_images = volume.shape[-1]

    cols = int(math.sqrt(n_images))
    rows = int(math.ceil(n_images / cols))

    image = []
    for r in range(rows):
        image.append([])
        for c in range(cols):
            if r * cols + c >= n_images:
                image[r].append(np.zeros(volume.shape[:-1]))
            else:
                image[r].append(volume[..., r * cols + c])

    for i in range(len(image)):
        image[i] = np.c_[tuple(image[i])]

    image = np.r_[tuple(image)]

    if len(plot_args) > 0:
        if 'figsize' in plot_args.keys():
            plt.figure(figsize=plot_args.pop('figsize'))
        plt.imshow(image, **plot_args)
        plt.show()

    return image


def modify(layer, index, model, target='hq'):
    modified = copy.copy(layer)
    modified[...,index] = np.zeros(layer.shape[:-1])
    #mosaic(modified)
    modified = np.reshape(modified, (1, *modified.shape))
    result = model.sess.run(model.hq.gen.layers[-1], {model.hq.gen.layers[-2]: modified})
    plt.imshow(result[0,...,0], cmap='inferno')
    plt.show()


def show(sample, min=None, max=None):
    temp = sample.copy()
    if min  is not None:
        vmin = min
        temp[temp<min] = 0
    else:
        vmin = temp.min()
    if max is not None:
        vmax = max
        temp[temp>max] = 0
    else:
        vmax = temp.max()
    plt.imshow(temp, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


