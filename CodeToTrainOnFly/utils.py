import tensorflow as tf
import copy
import numpy as np
from enum import Enum


class Layer(object):
    # simple class that stores a function and its parameters and is also callable

    def __init__(self, layer_func, var_scope='Layer', update_scope=True, training=None,
                 init=tf.contrib.layers.xavier_initializer, **params):
        """

        :param layer_func: function that is used on the input tensor
        :param var_scope: tf.var_scope in which the layer will be build
        :param update_scope: Specifies whether var_scope should be updated after each call by appending a number. Default: True
        :param training: training bool that is passed to the layer function, included because batch_norm requires it
        :param init: initialization method for the layer weights
        :param params: keyword arguments that are passed to layer_func
        """
        self.layer_func = layer_func
        self.training = training
        self.init = init
        self.params = {}
        for key, value in params.items():
            self.params[key] = value
        self.params['kernel_initializer'] = self.init
        self.params['bias_initializer'] = self.init
        self._initial_params = copy.deepcopy(self.params)
        if training is not None: self.params['training'] = training
        self.var_scope = var_scope
        self.update_scope = update_scope
        if update_scope: self.var_scope += '_0'
        self._call_count = 0

    def __call__(self, input, *args, **kwargs):
        """
        Build the layer in the tensorflow graph
        :param input: inuput tensor to the layer
        :return: output tensor of the layer
        """

        # do not store called initializer function in the object, otherwise Layer is not picklable
        params = copy.copy(self.params)
        params['kernel_initializer'] = self.params['kernel_initializer']()
        params['bias_initializer'] = self.params['bias_initializer']()

        with tf.variable_scope(self.var_scope):
            self._call_count += 1
            if self.update_scope:
                self.var_scope = self.var_scope[:-len(str(self._call_count)) - 1] + '_{}'.format(self._call_count)
            return self.layer_func(input, **params)

    def reset_params(self):
        """
        reset the parameters of the Layer object
        """
        self.params = copy.deepcopy(self._initial_params)
        if self.training is not None: self.params['training'] = self.training

    def reset_count(self):
        """
        reset the var_scope and call_count of the Layer object
        """
        if self.update_scope:
            self.var_scope = self.var_scope[:-len(str(self._call_count)) - 1] + '_0'
            self._call_count = 0

    def reset(self):
        """
        reset parameters, call_count and var_scope
        """
        self.reset_params()
        self.reset_count()


class SkipConLayer(Layer):
    """
    similiar to Layer, except that it stores the index of the layer that it should have a skip connection to
    """

    def __init__(self, layer_func, idx, var_scope='SkipConLayer', update_scope=True, training=None,
                 init=tf.contrib.layers.xavier_initializer, **params):
        """

        :param layer_func: function that is used on the input tensor
        :param idx: index of the layer that the skip connection should connect to
        :param var_scope: tf.var_scope in which the layer will be build
        :param update_scope: Specifies whether var_scope should be updated after each call by appending a number. Default: True
        :param training: training bool that is passed to the layer function, included because batch_norm requires it
        :param init: initialization method for the layer weights
        :param params: keyword arguments that are passed to layer_func
        """
        super().__init__(layer_func, var_scope, update_scope, training, init, **params)
        self.idx = idx

    def __call__(self, inputs, *args, **kwargs):
        """
        Build the layer in the tensorflow graph
        :param input: inuput tensor to the layer and the output layer of the layer idx
        :return: output tensor of the layer
        """

        params = copy.copy(self.params)
        params['kernel_initializer'] = self.params['kernel_initializer']()
        params['bias_initializer'] = self.params['bias_initializer']()

        with tf.variable_scope(self.var_scope):
            self._call_count += 1
            if self.update_scope:
                self.var_scope = self.var_scope[:-len(str(self._call_count)) - 1] + '_{}'.format(self._call_count)

            # concatenate the two input tensors along the channel dimension
            concat = tf.concat(inputs, 3)

            return self.layer_func(concat, **params)


class ResBlock(object):
    """
    residual block: one path goes through convolution ops the other is identity. both are added in the end.
    """

    def __init__(self, layer, n_layers, activation, var_scope='ResBlock', update_count=True, **params):
        """

        :param layer: Layer object that is used as convolution op
        :param n_layers: number of times the layer is called in the residual block
        :param activation: activation function applied after summation of tensors from convolution and identity paths
        :param var_scope: tf.var_scope in which the block is build. Default: 'ResBlock'
        :param update_count: Specifies whether var_scope should be updated after each call by appending a number. Default: True
        :param params: keyword arguments for the activation function
        """
        self.layer = layer
        self.n_layers = n_layers
        self.activation = activation
        self.var_scope = var_scope
        self.update_count = update_count
        if update_count: self.var_scope += '_0'
        self.__call_count = 0
        self.params = {}
        for key, value in params.items():
            self.params[key] = value

    def __call__(self, input, *args, **kwargs):
        """
        builds the residual block in the tensorflow graph
        :param input: input tensor to the block
        :return: list of output tensors at each layer in the block
        """
        # check to make sure that addition at the end of the block works
        assert input.shape[-1] == self.layer.params['filters'] or input.shape[-1] == 1

        # build a residual block invoking the passed layer n_layers times
        with tf.variable_scope(self.var_scope):
            layers = [input]
            for _ in range(self.n_layers - 1):
                layers.append(self.layer(layers[-1]))

            # add last layer with act_func after adding identity
            self.layer.params['activation'] = None
            layers.append(self.layer(layers[-1]))
            self.layer.reset()
            layers.append(self.activation((input + layers[-1]), **self.params))

        # remove input and batchnorm layer from list of built layers
        layers.pop(0)
        layers.pop(-2)

        # update var_scope
        self.__call_count += 1
        if self.update_count:
            self.var_scope = self.var_scope[:-len(str(self.__call_count)) - 1] + '_{}'.format(self.__call_count)

        return layers

    def reset(self):
        """
        resets own var_scope and call_count
        resets the Layer object's parameters, call_count and var_scope
        """
        if self.update_count:
            self.var_scope = self.var_scope[:-len(str(self.__call_count)) - 1] + '_0'
            self.__call_count = 0
        self.layer.reset()

    def reset_count(self):
        """
        resets own var_scope and call_count
        resets the Layer object's call_count and var_scope
        """
        if self.update_count:
            self.var_scope = self.var_scope[:-len(str(self.__call_count)) - 1] + '_0'
            self.__call_count = 0
        self.layer.reset_count()


class ImagePool(object):
    """
    image pool class for storing generated samples in the tensorflow graph
    """

    def __init__(self, pool_size, input_size, dtype=tf.float32, name='ImagePool', graph=tf.get_default_graph()):
        """

        :param pool_size: max number of samples in the pool
        :param input_size: dimensions of samples
        :param dtype: dtype of samples
        :param name: name string of the ImagePool
        :param graph: tensorflow graph in which the pool will be build
        """

        self.batch_size = input_size[0]
        assert pool_size > self.batch_size
        self.pool_size = pool_size
        self.input_size = input_size
        self.dtype = dtype
        self.graph = graph

        with tf.name_scope(name) as self.scope:
            # initialize the pool as a tf.Variable with random noise as samples
            self.pool = tf.Variable(np.random.randn(self.pool_size, *self.input_size[1:]),
                                    dtype=dtype, name='Pool', trainable=False)

            # random nodes for writing probability and the index in the pool
            self.rand_ind = tf.random_uniform((), 0, self.pool_size-self.batch_size, tf.int32)
            self.prob = tf.random_uniform((), 0, 1, tf.float16)

            # constant provided as fail path in tf.cond in write
            self.dummy_const = tf.constant(0, dtype=dtype)

    # build read and write nodes
    def write(self, tensor):
        """
        write tensor to a random index in the pool if random check is passed
        :param tensor: tensor to write
        :return: write op in the tensorflow graph
        """
        with tf.name_scope(self.scope):
            with tf.name_scope('write'):
                write_op = self.pool[self.rand_ind:self.rand_ind+self.batch_size].assign(tensor)
                write_op = self.pool.assign(tf.random_shuffle(write_op))
            return write_op

    def read(self):
        """
        reads a sample from a random location in the pool
        :return: read op in the tensorflow graph
        """
        with tf.name_scope(self.scope):
            with tf.name_scope('read'):
                read_op = self.pool[self.rand_ind:self.rand_ind+self.batch_size]
            return read_op


class InputMode(Enum):
    """
    helper enum for the switching feed and dataset mode
    """
    FEED = 1
    DATASET = 2


class DebugOptions(object):
    """helper class for specifying which summaries should be written during training"""

    def __init__(self, graph=False, runtime=False, hist=False):
        """

        :param graph: if True the tensorflow graph will be written to the eventsfile
        :param runtime: if True runtime statistics will be written to the eventsfile, has no effect if graph is False
        :param hist: if True histogramms of the network weights will be written to the eventsfile
        """

        self.graph = graph
        self.hist = hist
        if graph:
            self.runtime = runtime
        else:
            self.runtime = False


class GANStructure(object):
    """
    simple class that provides structure for GAN variables
    """

    class Gen(object):

        def __init__(self):
            self.from_real = None
            self.cyc = None
            self.layers = []
            self.loss = None
            self.trainer = None
            self.vars = None

    class Dis(object):

        def __init__(self):
            self.real = None
            self.fake = None
            self.pool_fake = None
            self.layers = []
            self.loss = None
            self.trainer = None
            self.vars = None

    def __init__(self, name):
        """
        :param name: name string that will be assigned to the class
        """

        self.name = name
        self.pool = None
        self.pool_write = None
        self.gen = self.Gen()
        self.dis = self.Dis()
        self.range_loss = None

        # init logging vars
        self.logdir = None
        self.summary_writer = None

    def __repr__(self):
        return self.name + ": " + super().__repr__()


def apply_activation(input, activation, **activation_params):
    """
    helper funciton that applies activation to an input tensor
    :param input: input tensor
    :param activation: function that is used on input
    :param activation_params: keyword arguments that are passed to the activation function
    :return: output tensor of the operation
    """

    if activation is not None:
        return activation(input, **activation_params)
    else:
        return input


def instance_norm(x, use_offset=False):
    """unlike batch norm, instance norm normalizes every tensor individually"""

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0)) if use_offset else 0
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset if use_offset else scale*tf.div(x-mean, tf.sqrt(var+epsilon))

    return out


def general_conv_layer(input, conv_func, activation, norm_func=instance_norm, activation_params=None, training=False, **params):
    """
    generalization of a convolution layer that includes norm before activation
    :param input: input tensor
    :param conv_func: convolution function
    :param activation: activation function
    :param norm_func: normalization function, defaults to instance_norm
    :param activation_params: keyword arguments passed to the activation function
    :param training: bool that tells the batch_norm whether it is in training mode
    :param params: keyword arguments passed to conv_func
    :return: output tensor of the layer
    """

    activation_params = {'name': None} if activation_params is None else activation_params

    # build convolution, norm and activation in that order
    layer = conv_func(input, **params)
    if norm_func is not None:
        opts = {}
        if norm_func == tf.layers.batch_normalization: opts = {'training': training}
        layer = norm_func(layer, **opts)

    return apply_activation(layer, activation, **activation_params)


def decision_layer(input, activation=None, activation_params=None, **params):
    """
    convolution layer that outputs a scalar for each sample in the batch
    :param input: input tensor. must be of size (None, 1, 1, 1)
    :param activation: activation function
    :param activation_params: keyword arguments passed to the activation function
    :param params: keyword arguments passed to the convolution
    :return: output tensor with scalar shape
    """

    activation_params = {'name': None} if activation_params is None else activation_params

    layer = tf.layers.conv2d(input, **params, use_bias=False)
    layer = apply_activation(layer, activation, **activation_params)
    assert all([x == 1 for x in layer.shape[1:]])
    return layer


def identity(input):
    return input


def relu1(tensor, name=None):
    return tf.minimum(tf.ones_like(tensor), tf.nn.relu(tensor), name=name)


def resize_conv(input, filters, kernel_size, strides=(1, 1), scale_factor=2, align_corners=False,
                padding='REFLECT', name='resize_conv2d', **params):
    """
    alternative to transpose convolutions that combines upscaling by interpolation with a padded convolution
    :param input: input tensor
    :param filters: number of channels after the convolution
    :param kernel_size: size of the convolution kernel
    :param strides: amount of stride in each dimension, defaults to (1,1)
    :param scale_factor: factor used for upscaling, defaults to 2
    :param align_corners: same as in tf.image.resize_bilinear
    :param padding: type of padding that should be used (see tf.pad), defaults to 'REFLECT'
    :param name: namescope for this function
    :param params: keyword arguments that will be passed to tf.layers.conv2d
    :return: output tensor of the convolution
    """
    with tf.name_scope(name):
        h, w = int(input.shape[1]), int(input.shape[2])
        h_n, w_n = [scale_factor * h, scale_factor * w]
        upscaled = tf.image.resize_bilinear(input, [h_n, w_n], align_corners)
        if padding is not None:
            h_o, w_o = (h_n-kernel_size + 1) / strides[1], (w_n-kernel_size + 1) / strides[1]
            paddings = (max(h_n - h_o, 0), max(w_n - w_o, 0))
            paddings = [[0, 0],
                        [np.floor(paddings[0]/2), paddings[0] - np.floor(paddings[0]/2)],
                        [np.floor(paddings[1]/2), paddings[1] - np.floor(paddings[1]/2)],
                        [0, 0]]
            paddings = np.array(paddings, np.int32)
            upscaled = tf.pad(upscaled, paddings, padding)
        out = tf.layers.conv2d(upscaled, filters, kernel_size, strides, padding='valid', **params)

    return out