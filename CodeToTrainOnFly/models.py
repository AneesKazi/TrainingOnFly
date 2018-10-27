import os
import numpy as np
import tensorflow as tf
from utils import Layer, SkipConLayer, ResBlock, DebugOptions, InputMode, ImagePool, GANStructure,\
    resize_conv, general_conv_layer, identity, decision_layer, instance_norm, relu1
import pickle
import argparse
#import cv2 as cv
import sys
from time import ctime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# noinspection PyShadowingNames,PyUnusedLocal
class CyclicGAN(object):

    def __init__(self, gen_arch, dis_arch, inputs, training, logdir=None, pool_size=50, learning_rate=0.0002,
                 loss_weights=(1, 1), graph=None, interval=1000, debug=DebugOptions(), n_saves=50, inference_mode=False):
        """
        builds a CyclicGAN object for training and inference
        :param gen_arch: generator architecture. either dict with build parameters (see static methods of this class) or list of Layer / ResBlock Objects
        :param dis_arch: discriminator architecture. either dict with build parameters (see static methods of this class) or list of Layer / ResBlock Objects
        :param inputs: inputs to the model. dict of either tf.placeholder or tf.data.Iterator objects marked as hq and lq
        :param training: tf.placeholder (bool type) object which sets the training flag for batch_norm layers
        :param logdir: string directory where to save logs to
        :param pool_size: int size of the image pool
        :param learning_rate: initial learning rate. linearly decreased to zero after 100 epochs
        :param loss_weights: weights for gan loss. first entry weights cycle loss, second discriminator loss
        :param graph: graph in which the model is build
        :param interval: number of training steps after which summaries are written
        :param debug: DebugOptions object that specifies what additional summaries should be recorded, default only images and scalars
        :param n_saves: number of checkpoints the saver will keep
        :param inference_mode: if True losses, summaries and trainers will not be built in the init. results in a more lightweight graph for inference
        """
        print('initializing model...')
        self.inference_mode = inference_mode
        tf.logging.set_verbosity(tf.logging.ERROR)
        print('\tinitializing variables...')
        self.debug = debug
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        self.sess = tf.Session(graph=self.graph)

        # infer input mode from inputs
        self.inputs = inputs
        self.real_hq, self.real_lq = None, None
        self.train_method, self.generate_method, self.discriminate_method = None, None, None
        if all([type(v) is tf.Tensor for v in inputs.values()]):
            self.mode = InputMode.FEED
            self.train_method = self._feed_training_epoch
            self.generate_method = self._generate_from_placeholder
            self.discriminate_method = self._discriminate_from_placeholder
            self.input_size = inputs['lq'].shape
            self.input_dtype = inputs['lq'].dtype
        elif all([type(v) is tf.data.Iterator for v in inputs.values()]):
            self.mode = InputMode.DATASET
            self.train_method = self._dataset_training_epoch
            self.generate_method = self._generate_from_iterator
            self.discriminate_method = self._discriminate_from_iterator
            self.input_size = inputs['lq'].output_shapes
            self.input_dtype = inputs['lq'].output_types
            self.input_pl = tf.placeholder(dtype=self.input_dtype, shape=tuple([*self.input_size]))
            self.input_set = tf.data.Dataset.from_tensors(self.input_pl)
        else:
            raise TypeError('inputs must be either tf.data.Iterator or tf.Tensor objects')

        self.interval = interval
        self.learning_rate = learning_rate
        self.lr_var = tf.Variable(initial_value=learning_rate, dtype=tf.float64, name='learning_rate', trainable=False)
        self.pool_size = pool_size
        self.training = training if not inference_mode else False
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        self.epoch = 0
        self.loss_weights = loss_weights
        self.train_op = None

        # parse network architecture
        print('\tparsing architectures...')
        self.gen_arch = self.parse_arch(gen_arch, self.generate_gen_arch)
        self.dis_arch = self.parse_arch(dis_arch, self.generate_dis_arch)

        # initialize GAN variables
        self.hq = GANStructure('HQ')
        self.lq = GANStructure('LQ')

        # init logging vars
        self.logdir = None
        self.summary_writer = None
        self.summaries = {}
        self.summary = None

        # set up the neural network
        print('\tbuilding net...')
        self._build_net()
        if not self.inference_mode:
            print('\tsetting up losses...')
            self._setup_losses()
            print('\tsetting up trainers...')
            self._setup_train_op()
            print('\tsetting up summaries...')
            self._setup_summaries()

        if not self.inference_mode:
            # init tf variables
            self.sess.run(tf.global_variables_initializer())
            if self.mode == InputMode.DATASET:
                self.sess.run([self.inputs['lq'].initializer, self.inputs['hq'].initializer])

            print('\tinitializing logging...')
            self.set_logdir(logdir)

        # initialize saver
        self.saver = tf.train.Saver(max_to_keep=n_saves)

        print('done!', '', sep='\n')

    def _build_net(self):
        """
        build the computational graph accordding to the specified architectures
        :return: None
        """

        with tf.variable_scope('cycleGAN') as scope:

            # prep input tensors
            if self.mode == InputMode.DATASET:
                self.real_hq = self.inputs['hq'].get_next()
                self.real_lq = self.inputs['lq'].get_next()
            else:
                self.real_hq = self.inputs['hq']
                self.real_lq = self.inputs['lq']

            # build branches that operate on inputs
            with tf.name_scope('LQ_Gen') as lq_gen_scope:
                self.lq.gen.layers = self.build_arch(self.real_hq, self.gen_arch, 'LQ_Gen', self.graph)
            with tf.name_scope('HQ_Gen') as hq_gen_scope:
                self.hq.gen.layers = self.build_arch(self.real_lq, self.gen_arch, 'HQ_Gen', self.graph)
            with tf.name_scope('LQ_Dis') as lq_dis_scope:
                self.lq.dis.layers = self.build_arch(self.real_lq, self.dis_arch, 'LQ_Dis', self.graph)
            with tf.name_scope('HQ_Dis') as hq_dis_scope:
                self.hq.dis.layers = self.build_arch(self.real_hq, self.dis_arch, 'HQ_Dis', self.graph)

            # store outputs of these branches
            self.lq.gen.from_real = self.lq.gen.layers[-1]
            self.hq.gen.from_real = self.hq.gen.layers[-1]
            self.lq.dis.real = self.lq.dis.layers[-1]
            self.hq.dis.real = self.hq.dis.layers[-1]

            # only build these parts in training mode
            if not self.inference_mode:

                # build image pools
                self.lq.pool = ImagePool(self.pool_size, self.input_size, name='LQ-Pool')
                self.hq.pool = ImagePool(self.pool_size, self.input_size, name='HQ-Pool')

                # re-build components so that generated images are also taken into account
                scope.reuse_variables()
                with tf.name_scope(lq_gen_scope):
                    self.lq.gen.cyc = self.build_arch(self.hq.gen.from_real, self.gen_arch, 'LQ_Gen', self.graph)[-1]
                with tf.name_scope(hq_gen_scope):
                    self.hq.gen.cyc = self.build_arch(self.lq.gen.from_real, self.gen_arch, 'HQ_Gen', self.graph)[-1]
                with tf.name_scope(lq_dis_scope):
                    self.lq.dis.pool_fake = self.build_arch(self.lq.pool.read(), self.dis_arch, 'LQ_Dis', self.graph)[-1]
                    scope.reuse_variables()
                    self.lq.dis.fake = self.build_arch(self.lq.gen.from_real, self.dis_arch, 'LQ_Dis', self.graph)[-1]
                with tf.name_scope(hq_dis_scope):
                    self.hq.dis.pool_fake = self.build_arch(self.hq.pool.read(), self.dis_arch, 'HQ_Dis', self.graph)[-1]
                    scope.reuse_variables()
                    self.hq.dis.fake = self.build_arch(self.hq.gen.from_real, self.dis_arch, 'HQ_Dis', self.graph)[-1]

                # build write ops to image pools
                self.lq.pool_write = self.lq.pool.write(self.lq.gen.from_real)
                self.hq.pool_write = self.hq.pool.write(self.hq.gen.from_real)

    def _setup_losses(self):
        """
        setup all loss functions
        :return: None
        """

        # generator losses = cycle consistency loss + discriminator output
        def gen_loss(cycle_loss, fake_dis, w, name='gen_loss'):
            with tf.variable_scope(name):
                discriminator_loss = tf.reduce_mean(tf.squared_difference(fake_dis, 1))
                generator_loss = w[0] * cycle_loss + w[1] * discriminator_loss

            return generator_loss

        # discriminator losses = (discrimination of fake(from pool) + discrimination of real)/2
        def dis_loss(fake_dis, real_dis, name='gen_loss'):
            with tf.variable_scope(name):
                fake_dis_loss = tf.reduce_mean(tf.square(fake_dis))
                real_dis_loss = tf.reduce_mean(tf.squared_difference(real_dis, 1.0))
                discriminator_loss = (fake_dis_loss + real_dis_loss) / 2.0

            return discriminator_loss

        # set up cycle consistency loss
        with tf.name_scope('cycleGAN/cycle_loss'):
            c_loss = tf.reduce_mean(tf.abs(self.lq.gen.cyc - self.real_lq)) + \
                     tf.reduce_mean(tf.abs(self.hq.gen.cyc - self.real_hq))

        # strong penalty for going over the max value or under min value
        with tf.name_scope('cycleGAN/max_loss'):
            upper_bound = 1
            lower_bound = 0
            lq_range_loss = tf.reduce_mean(tf.nn.relu(self.lq.gen.from_real - upper_bound))
            lq_range_loss += tf.reduce_mean(tf.nn.relu(self.lq.gen.cyc - upper_bound))
            lq_range_loss += tf.reduce_mean(tf.nn.relu(lower_bound - self.lq.gen.from_real))
            lq_range_loss += tf.reduce_mean(tf.nn.relu(lower_bound - self.lq.gen.cyc))
            hq_range_loss = tf.reduce_mean(tf.nn.relu(self.hq.gen.from_real - upper_bound))
            hq_range_loss += tf.reduce_mean(tf.nn.relu(self.hq.gen.cyc - upper_bound))
            hq_range_loss += tf.reduce_mean(tf.nn.relu(lower_bound - self.hq.gen.from_real))
            hq_range_loss += tf.reduce_mean(tf.nn.relu(lower_bound - self.hq.gen.cyc))

            self.hq.range_loss = hq_range_loss
            self.lq.range_loss = lq_range_loss

        # set up losses for each of the 4 model parts
        max_loss_weight = 0.1
        self.lq.gen.loss = gen_loss(c_loss, self.lq.dis.fake, self.loss_weights, name='lq_gen_loss') + max_loss_weight * self.lq.range_loss
        self.hq.gen.loss = gen_loss(c_loss, self.hq.dis.fake, self.loss_weights, name='hq_gen_loss') + max_loss_weight * self.hq.range_loss
        self.lq.dis.loss = dis_loss(self.lq.dis.pool_fake, self.lq.dis.real, name='lq_dis_loss')
        self.hq.dis.loss = dis_loss(self.hq.dis.pool_fake, self.hq.dis.real, name='hq_dis_loss')

    def _setup_train_op(self):
        """
        build the training ops for each model part in the tensorflow graph
        :return: None
        """

        # collect variables for the seperate modules
        trainable_variables = tf.trainable_variables()

        self.lq.gen.vars = [var for var in trainable_variables if 'LQ_Gen' in var.name]
        self.hq.gen.vars = [var for var in trainable_variables if 'HQ_Gen' in var.name]
        self.lq.dis.vars = [var for var in trainable_variables if 'LQ_Dis' in var.name]
        self.hq.dis.vars = [var for var in trainable_variables if 'HQ_Dis' in var.name]

        self.optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=0.5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            # global step is only updated in the last training op, otherwise it would jump in blocks of 4 at every training step
            # training ops for the generators are grouped with the write ops for their respective image pools
            self.lq.gen.trainer = self.optimizer.minimize(self.lq.gen.loss, var_list=self.lq.gen.vars, name='LQ_Gen_Trainer')
            self.lq.gen.trainer = tf.group(self.lq.gen.trainer, self.lq.pool_write)

            self.hq.gen.trainer = self.optimizer.minimize(self.hq.gen.loss, var_list=self.hq.gen.vars, name='HQ_Gen_Trainer')
            self.hq.gen.trainer = tf.group(self.hq.gen.trainer, self.hq.pool_write)

            self.lq.dis.trainer = self.optimizer.minimize(self.lq.dis.loss, var_list=self.lq.dis.vars, name='LQ_Dis_Trainer')

            self.hq.dis.trainer = self.optimizer.minimize(self.hq.dis.loss, global_step=self.global_step,
                                                          var_list=self.hq.dis.vars, name='HQ_Dis_Trainer')

            self.train_op = tf.group([self.lq.gen.trainer, self.hq.gen.trainer, self.lq.dis.trainer, self.hq.dis.trainer])

    def _setup_summaries(self):
        """
        build all necessary summary ops for tensorboard visualization
        :return: None
        """

        with tf.name_scope('summaries'):

            with tf.name_scope('losses'):
                # set loss summaries
                self.summaries['lq-gen-loss'] = tf.summary.scalar('LQ-Generator-Loss', tf.reduce_sum(self.lq.gen.loss))
                self.summaries['lq-dis-loss'] = tf.summary.scalar('LQ-Discriminator-Loss', tf.reduce_sum(self.lq.dis.loss))
                self.summaries['hq-gen-loss'] = tf.summary.scalar('HQ-Generator-Loss', tf.reduce_sum(self.hq.gen.loss))
                self.summaries['hq-dis-loss'] = tf.summary.scalar('HQ-Discriminator-Loss', tf.reduce_sum(self.hq.dis.loss))
                self.summaries['lq-range-loss'] = tf.summary.scalar('LQ-Range-Loss', tf.reduce_sum(self.lq.range_loss))
                self.summaries['hq-range-loss'] = tf.summary.scalar('HQ-Range-Loss', tf.reduce_sum(self.hq.range_loss))
                self.summaries['losses'] = tf.summary.merge([s[1] for s in self.summaries.items() if 'loss' in s[0]])

            with tf.name_scope('discriminator_outputs'):
                # set discriminator output summaries
                self.summaries['lq-dis-fake'] = tf.summary.scalar('LQ-Discriminator-Fake', tf.reduce_mean(self.lq.dis.fake))
                self.summaries['lq-dis-pool_fake'] = tf.summary.scalar('LQ-Discriminator-Pool_Fake', tf.reduce_mean(self.lq.dis.pool_fake))
                self.summaries['lq-dis-real'] = tf.summary.scalar('LQ-Discriminator-Real', tf.reduce_sum(self.lq.dis.real))
                self.summaries['hq-dis-fake'] = tf.summary.scalar('HQ-Discriminator-Fake', tf.reduce_mean(self.hq.dis.fake))
                self.summaries['hq-dis-pool_fake'] = tf.summary.scalar('HQ-Discriminator-Pool_Fake', tf.reduce_mean(self.hq.dis.pool_fake))
                self.summaries['hq-dis-real'] = tf.summary.scalar('HQ-Discriminator-Real', tf.reduce_sum(self.hq.dis.real))
                self.summaries['dis-outputs'] = tf.summary.merge([s[1] for s in self.summaries.items() if 'fake' in s[0] or 'real' in s[0]])

            with tf.name_scope('gradients'):
                # set gradient summaries

                lq_gen_grads = [grad for grad in self.optimizer.compute_gradients(self.lq.gen.loss) if grad[0] is not None]
                hq_gen_grads = [grad for grad in self.optimizer.compute_gradients(self.hq.gen.loss) if grad[0] is not None]
                lq_dis_grads = [grad for grad in self.optimizer.compute_gradients(self.lq.dis.loss) if grad[0] is not None]
                hq_dis_grads = [grad for grad in self.optimizer.compute_gradients(self.hq.dis.loss) if grad[0] is not None]

                self.summaries['lq-gen-grads'] = tf.summary.scalar('LQ-Gen-Gradients',  tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t in lq_gen_grads]))
                self.summaries['hq-gen-grads'] = tf.summary.scalar('HQ-Gen-Gradients', tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t in hq_gen_grads]))
                self.summaries['lq-dis-grads'] = tf.summary.scalar('LQ-Dis-Gradients', tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t in lq_dis_grads]))
                self.summaries['hq-dis-grads'] = tf.summary.scalar('HQ-Dis-Gradients', tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t in hq_dis_grads]))
                self.summaries['gradients'] = tf.summary.merge([s[1] for s in self.summaries.items() if 'grads' in s[0]])

            # combine summaries that should be run at every step
            self.summaries['common'] = tf.summary.merge([self.summaries['gradients'], self.summaries['dis-outputs'], self.summaries['losses']])

            # set image pool summaries
            def parse_pool_to_summ(pool, name, max_im=None):
                """
                small helper for concating pool tensors into one tensor and creating summary
                :param pool: ImagePool object
                :param name: str name of the summary
                :param max_im: summary max_outputs argument
                :return: image summary
                """
                with tf.variable_scope(name):
                    pool_ims = [pool[i] for i in range(pool.shape[0])]
                    pool_tensor = tf.concat(pool_ims, 0)
                    max = max_im if max_im is not None else pool_tensor.shape[0]
                    summ = tf.summary.image(name, pool_tensor, max)

                return summ

            self.summaries['hq-image-pool'] = tf.summary.image('HQ-Pool', self.hq.pool.pool, 3, family='HQ')
            self.summaries['lq-image-pool'] = tf.summary.image('LQ-Pool', self.lq.pool.pool, 3, family='LQ')
            self.summaries['pools'] = tf.summary.merge([s[1] for s in self.summaries.items() if 'pool' in s[0]])

            # only log histograms if the according debug flag is set
            if self.debug.hist:
                # set variable summaries
                def parse_vars_to_summ(vars):
                    """
                    helper for creating a hist summary from trainable variables
                    :param vars: list of tf.Variables
                    :return: histogram summary
                    """
                    var_names = [str(v).split('\'')[1].replace(':', '_') for v in vars]
                    summaries = [tf.summary.histogram(vn, v) for vn, v in zip(var_names, vars)]
                    return tf.summary.merge(summaries)

                self.summaries['lq-gen-vars'] = parse_vars_to_summ(self.lq.gen.vars)
                self.summaries['lq-dis-vars'] = parse_vars_to_summ(self.lq.dis.vars)
                self.summaries['hq-gen-vars'] = parse_vars_to_summ(self.hq.gen.vars)
                self.summaries['hq-dis-vars'] = parse_vars_to_summ(self.hq.dis.vars)
                self.summaries['hists'] = tf.summary.merge([s[1] for s in self.summaries.items() if 'vars' in s[0]])
                self.summaries['epoch'] = tf.summary.merge([self.summaries['pools'], self.summaries['hists']])
            else:
                self.summaries['epoch'] = self.summaries['pools']

    @staticmethod
    def parse_arch(arch, build_method):
        """
        parses arch so that the output is always a list of Layer and/or ResBlock objects
        :param arch: architecture for the network either as dict with build parameters or list of Layer objects
        :param build_method: method to invoke when arch is a dict
        :return: list of Layer and/or ResBlock objects
        """
        if type(arch) is dict:
            return build_method(**arch)
        else:
            return arch

    @staticmethod
    def generate_gen_arch(scale_factor, n_residuals, n_reslayers, activation=tf.nn.relu, input_channels=1, filters=8,
                          kernel_size=4, norm_func=instance_norm, training=False, skip_conn=False,
                          down_conv=tf.layers.conv2d, up_conv=resize_conv, **activation_params):
        """
        helper function for building generator architectures
        :param scale_factor: number of down / up scaling layers
        :param n_residuals: number of residual blocks
        :param n_reslayers: number of layers in each residual block
        :param activation: activation function
        :param input_channels: number of channels that the input tensors will have
        :param filters: number of channels after the initial convolution layer
        :param kernel_size: kernel size for all convolutions except the initial one, which is currently set to 7
        :param norm_func: normalization in the network
        :param training: training flag, only useful when using batch norm
        :param skip_conn: if True, skip connections will be added between down and up scaling layers
        :param down_conv: convolution that will be used for all layers except upscaling
        :param up_conv: convolution that will be used in the upscaling layers
        :param activation_params: keyword arguments for the activation function
        :return: list of (SkipCon)Layer and ResBlock objects that specifies the generator architecture
        """

        # initialize parameters for the layer objects
        params = {'conv_func': down_conv, 'activation': activation, 'training': training,
                  'filters': filters, 'kernel_size': 7, 'padding': 'same', 'norm_func': norm_func}

        # initial 7x7 convolution
        var_scope = 'initial_conv_{}'.format(params['filters'])
        layers = [Layer(general_conv_layer, var_scope=var_scope, update_scope=False, **params)]

        # downsampling with strided convolutions, channels are doubled with every downsample
        params['kernel_size'] = kernel_size
        params['strides'] = (2, 2)
        skip_idx = []
        for i in range(scale_factor):
            params['filters'] *= 2
            var_scope = 'd_conv_{}'.format(params['filters'])
            layers.append((Layer(general_conv_layer, var_scope=var_scope, update_scope=False, **params)))
            if skip_conn:
                skip_idx.append(len(layers))

        # add n_residual residual blocks
        params['strides'] = (1, 1)
        var_scope = 'gen_conv_{}'.format(params['filters'])
        conv_layer = Layer(general_conv_layer, var_scope=var_scope, **params)
        res_block = ResBlock(conv_layer, n_reslayers, activation, **activation_params)
        res_blocks = [res_block] * n_residuals
        layers += res_blocks

        # upsampling with fractionaly strided convolutions, channels are halved with every upsampling
        # params['strides'] = (2, 2)
        params['conv_func'] = up_conv
        padding = params.pop('padding')
        for i in range(scale_factor-1, -1, -1):
            params['filters'] //= 2
            var_scope = 'u_conv_{}'.format(params['filters'])
            if skip_conn:
                layers.append((SkipConLayer(general_conv_layer, skip_idx[i], var_scope, update_scope=False, **params)))
            else:
                layers.append((Layer(general_conv_layer, var_scope, update_scope=False, **params)))

        # add final convolution to end up with initial number of channels
        # params['strides'] = (1, 1)
        params['conv_func'] = down_conv
        params['filters'] = input_channels
        params['padding'] = padding
        params['activation'] = activation
        params['norm_func'] = None
        var_scope = 'final_conv_{}'.format(params['filters'])
        layers.append(Layer(general_conv_layer, var_scope=var_scope, update_scope=False, **params))

        return layers

    @staticmethod
    def generate_dis_arch(input_dim=512, activation=tf.nn.leaky_relu, filters=1, training=False, patch=None,
                          norm_func=instance_norm, **activation_params):
        """
        helper function for building discriminator architectures
        :param input_dim: dimensions of the inputs, square images are assumed
        :param activation: activation function in the network
        :param filters: number of channels after the first convolution
        :param training: training flag, only useful when using batch norm
        :param patch: size of the quadratic patch that will be used as input, switches to patchGAN mode and overrides input_dim
        :param norm_func: normalization in the network
        :param activation_params: keyword arguments for the activation function
        :return: list of Layer objects that specifies the discriminator architecture
        """

        # initialize parameters for the layer objects
        if not activation_params: activation_params['name'] = None
        params = {'conv_func': tf.layers.conv2d, 'activation': activation, 'training': training,
                  'filters': filters, 'kernel_size': 4, 'padding': 'same', 'strides': (2, 2),
                  'norm_func': norm_func}
        layers = []

        # random cropping if running in patchGAN mode
        if patch:
            layers.append(Layer(tf.random_crop, size=patch))
            layers[-1].params.pop('kernel_initializer')
            layers[-1].params.pop('bias_initializer')
            input_dim = patch[1]

        n_layers = int(np.log2(input_dim))

        # build specified number of downsampling conv_layers
        # downsampling with strided 4x4 convolutions, channels are doubled with every downsample
        for i in range(n_layers):

            # make filter smaller at the end of the loop
            if n_layers - i == 2:
                params['kernel_size'] = 2
                params['padding'] = 'valid'
            if n_layers - i == 1:
                params['strides'] = (1, 1)
                params['norm_func'] = None

            params['filters'] *= 2

            var_scope = 'd_conv_{}'.format(params['filters'])
            layers.append(Layer(general_conv_layer, var_scope=var_scope, activation_params=activation_params,
                                update_scope=False, **params))

        # add final convolution to produce scalar between 0 and 1
        params = {'filters': 1, 'kernel_size': 1, 'activation': None}
        layers.append(Layer(decision_layer, var_scope='decision_layer', update_scope=False, **params))

        return layers

    @staticmethod
    def build_arch(input, arch, name, graph=None):
        """
        build the architecture
        :param input: input tensor to the network
        :param arch: list of callable Layer objects
        :param name: name of the network
        :param graph: graph in which to build the network
        :return: list of the output tensors for each layer in the network
        """
        if graph is None: graph = tf.get_default_graph()
        with graph.as_default():
            with tf.variable_scope(name, auxiliary_name_scope=False):
                layers = [input]
                for layer in arch:
                    if type(layer) is Layer:
                        layers.append(layer(layers[-1]))
                    elif type(layer) is ResBlock:
                        layers += layer(layers[-1])
                    elif type(layer) is SkipConLayer:
                        layers.append(layer([layers[-1], layers[layer.idx]]))

                for layer in arch: layer.reset_count()
                return layers

    def set_logdir(self, logdir):
        """
        helper function for setting the path for log files that also automatically updates FileWriter

        Parameters
        ----------
        logdir : string
            path where all logs and checkpoints will be saved
        """

        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        if self.debug.graph:
            graph = self.graph
        else:
            graph = None

        self.logdir = logdir
        self.summary_writer = tf.summary.FileWriter(logdir, graph)

        # save architecture to logdir
        if not self.inference_mode:
            pickle.dump(self.gen_arch, open(os.path.join(logdir, 'generator_architecture.pkl'), 'wb'))
            pickle.dump(self.dis_arch, open(os.path.join(logdir, 'discriminator_architecture.pkl'), 'wb'))

    def _feed_training_epoch(self, feed_batch, *args, **kwargs):

        raise NotImplementedError('training using the feed mechanism is currently not supported')

    def train(self, n_epochs, decay_threshold, *args, **kwargs):
        """
        train the network for a set number of epochs
        :param n_epochs: number of epochs the network will be trained for
        :param decay_threshold: epoch after which the learning rate will start to decay
        :param args: positional arguments passed to the training method
        :param kwargs: keyword arguments passed to the training method
        :return: None
        """

        decay_threshold += self.epoch
        diff = n_epochs - decay_threshold

        for _ in range(n_epochs):

            # linearly decay learning rate after set amount of epochs
            if self.epoch > decay_threshold:
                lr_delta = self.learning_rate*(self.epoch - decay_threshold)/diff
                self.sess.run(self.lr_var.assign(self.learning_rate-lr_delta))

            # execute training for one epoch
            self.train_method(*args, **kwargs)

    def _dataset_training_step(self):
        """
        helper function that performs a single training step
        :return: True if iterator is exhausted, False otherwise
        """

        def run_train_ops(train_op, summary=None, metadata=False):
            if metadata:
                # init runtime stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                params = {'options': run_options, 'run_metadata': run_metadata}
            else:
                run_metadata = None
                params = {}
            _, summ_str = self.sess.run([train_op, summary], **params)
            self.write_summary(summ_str, run_metadata)

        try:
            # perform single training step and record losses
            step = self.sess.run(self.global_step)

            if step % (self.interval*25) == 0:
                summary = self.sess.run(self.summaries['epoch'])
                self.write_summary(summary)

            # record variable histograms at specified intervals
            if step % self.interval == 0:
                print('{}\tepoch: {}\tstep: {}'.format(ctime(), self.epoch, step))
                metadata = False
                if self.debug.runtime: metadata = True
                run_train_ops(self.train_op, self.summaries['common'], metadata)
            else:
                self.sess.run(self.train_op)

        except tf.errors.OutOfRangeError:
            # pass True to exit loop in dataset_training_epoch
            return True
        else:
            return False

    def _dataset_training_epoch(self, *args, **kwargs):
        """
        helper for training for one epoch
        :param args: not used
        :param kwargs: not used
        :return: None
        """

        # re-initialize iterator for this epoch
        self.sess.run([self.inputs['lq'].initializer, self.inputs['hq'].initializer])

        # loop until iterator is exhausted
        exhausted = False
        while not exhausted:
            exhausted = self._dataset_training_step()

        # record images from the pools
        summary = self.sess.run(self.summaries['epoch'])
        self.write_summary(summary)
        lr = self.sess.run(self.lr_var)
        print('epoch: {}\tlr {}'.format(self.epoch, lr))
        self.save()
        self.epoch += 1
        sys.stdout.flush()

    def write_summary(self, summary_str, metadata=None):
        """
        wrapper function for writing summaries
        summary_str : summary protocol buffer summary to be added to the event file
        """
        step = self.sess.run(self.global_step)
        if metadata is not None: self.summary_writer.add_run_metadata(metadata, str(step))
        self.summary_writer.add_summary(summary_str, step)
        self.summary_writer.flush()

    def save(self):
        """
        wrapper function for saving the model
        """
        self.saver.save(self.sess, os.path.join(self.logdir, "cGAN.ckpt"), global_step=self.global_step)

    def load(self, model_path):
        """
        wrapper function for restoring model weights
        :param model_path: str
            path to checkpoint file
        """
        print('loading checkpoint {}...'.format(model_path.split('/')[-1]))
        self.saver.restore(self.sess, model_path)
        print('done!', '', sep='\n')

    def discriminate(self, sample, target_domain):
        """
        get discriminator output for this sample
        :param sample: input sample as numpy 4D numpy array (batch size, height, width, channels)
        :param target_domain: 'hq' or 'lq' to specify which discriminator is targeted
        :return: output od the discriminator
        """

        if not self.inference_mode: self.sess.run(tf.assign(self.training, False))

        # sort out inputs
        if target_domain.lower() == 'lq':
            input = self.inputs['lq']
            tensor = self.lq.dis.real
        elif target_domain.lower() == 'hq':
            input = self.inputs['hq']
            tensor = self.hq.dis.real
        else:
            raise ValueError

        if not self.inference_mode: self.sess.run(tf.assign(self.training, True))

        return self.discriminate_method(sample, input, tensor)

    def generate(self, sample, target_domain):
        """
        generate an image from the sample
        :param sample: input sample as numpy 4D numpy array (batch size, height, width, channels)
        :param target_domain: 'hq' or 'lq' to specify which generator is targeted
        :return: output of the generator
        """

        if not self.inference_mode: self.sess.run(tf.assign(self.training, False))
        if target_domain.lower() == 'lq':
            input = self.inputs['hq']
            tensor = self.lq.gen.from_real
        elif target_domain.lower() == 'hq':
            input = self.inputs['lq']
            tensor = self.hq.gen.from_real
        else:
            raise ValueError

        if not self.inference_mode: self.sess.run(tf.assign(self.training, True))

        return self.generate_method(sample, input, tensor)

    def _generate_from_iterator(self, sample, input, tensor):

        init_op = input.make_initializer(self.input_set)
        self.sess.run(init_op, {self.input_pl: sample})
        result = self.sess.run(tensor)
        self.sess.run(input.initializer)

        return result

    def _generate_from_placeholder(self, sample,  input, tensor):

        result = self.sess.run(tensor, {input: sample})
        return result

    def _discriminate_from_placeholder(self, sample,  input, tensor):

        result = self.sess.run(tensor, {input: sample})
        return result

    def _discriminate_from_iterator(self, sample,  input, tensor):

        init_op = input.make_initializer(self.input_set)
        self.sess.run(init_op, {self.input_pl: sample})
        result = self.sess.run(tensor)
        self.sess.run(input.initializer)

        return result

    @classmethod
    def from_trained_model(cls, inputs, gen_arch, dis_arch, checkpoint):
        """
        factory method for building models from pickled architectures and a checkpoint file
        :param inputs: input tensors to the network
        :param gen_arch: pickled generator architecture
        :param dis_arch: pickled dicriminator architecture
        :param checkpoint: tensorflow chekpoint file
        :param logdir: directory for logging
        :return: loaded model in inference-mode
        """

        gen_arch = pickle.load(open(gen_arch, 'rb'))
        dis_arch = pickle.load(open(dis_arch, 'rb'))
        graph = tf.get_default_graph()

        cgan = CyclicGAN(gen_arch, dis_arch, inputs, None, None, graph=graph, inference_mode=True)
        cgan.load(checkpoint)

        return cgan


def setup():
    import time
    import cv2 as cv
    import os

    def build_datasets(folder, limit=256, batch=4):
        files = list(os.walk(folder))[0][2]
        np.random.shuffle(files)
        files = files[:limit]
        images = np.array([cv.imread(os.path.join(folder, f), 1) for f in files], np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(images).apply(tf.contrib.data.batch_and_drop_remainder(batch))
        dataset = dataset.repeat(1).shuffle(10).prefetch(batch)

        return dataset

    graph = tf.Graph()
    logdir = '/home/kazuki/testing/'
    with graph.as_default():
        with tf.name_scope('Dataset'):
            print('loading datasets...')
            start = time.time()
            dA = build_datasets('./ukiyoe2photo/trainA')
            dB = build_datasets('./ukiyoe2photo/trainB')
            print('duration: {}'.format(round(time.time() - start, 3)))
            inputs = {'lq': dA.make_initializable_iterator(),
                      'hq': dB.make_initializable_iterator()}

        print('building gan...')
        g_arch = CyclicGAN.generate_gen_arch(2, 3, 2, tf.nn.selu, skip_conn=True, input_channels=3, training=False, norm_func=identity)
        d_arch = CyclicGAN.generate_dis_arch(256, tf.nn.selu, training=False, norm_func=identity)
        cgan = CyclicGAN(g_arch, d_arch, inputs, None, logdir, graph=graph, inference_mode=True)
        # cgan.train(5, 3)

        print('duration: {}'.format(round(time.time() - start, 3)))

        return cgan, inputs, dA, dB


def main_test():
    import time
    import cv2 as cv
    import os

    def build_datasets(folder, limit=256, batch=4):
        files = list(os.walk(folder))[0][2]
        np.random.shuffle(files)
        files = files[:limit]
        images = np.array([cv.imread(os.path.join(folder, f), 1) for f in files], np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(images).apply(tf.contrib.data.batch_and_drop_remainder(batch))
        dataset = dataset.repeat(1).shuffle(10).prefetch(batch)

        return dataset

    graph = tf.Graph()
    logdir = '/home/kazuki/testing/'
    with graph.as_default():
        with tf.name_scope('Dataset'):
            print('loading datasets...')
            start = time.time()
            dA = build_datasets('./ukiyoe2photo/trainA')
            dB = build_datasets('./ukiyoe2photo/trainB')
            print('duration: {}'.format(round(time.time() - start, 3)))
            inputs = {'lq': dA.make_initializable_iterator(),
                      'hq': dB.make_initializable_iterator()}

        print('building gan...')
        g_arch = CyclicGAN.generate_gen_arch(2, 3, 2, tf.nn.selu, skip_conn=True, input_channels=3, training=False, norm_func=instance_norm)
        d_arch = CyclicGAN.generate_dis_arch(256, tf.nn.selu, training=False, norm_func=instance_norm)
        cgan = CyclicGAN(g_arch, d_arch, inputs, None, logdir, graph=graph, interval=1, debug=DebugOptions(), learning_rate=0.0002,
                         loss_weights=(1,10), n_saves=1)

        #cgan.train(5, 3)

        print('duration: {}'.format(round(time.time() - start, 3)))

        return cgan, inputs, dA, dB


def train_disc(steps=128):

    import time
    import cv2 as cv
    import os

    def build_datasets(folder, limit=10):
        files = list(os.walk(folder))[0][2][:limit]
        images = np.array([cv.imread(os.path.join(folder, f), 1) for f in files], np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(50).batch(16).repeat()

        return dataset

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Dataset'):
            print('loading datasets...')
            start = time.time()
            dA = build_datasets('./ukiyoe2photo/trainA')
            dB = build_datasets('./ukiyoe2photo/trainB')
            print('duration: {}'.format(round(time.time() - start, 3)))
            inputs = {'lq': dA.make_initializable_iterator(),
                      'hq': dB.make_initializable_iterator()}

        print('building discriminator...')
        with tf.Session(graph=graph) as sess:
            sess.run([inputs['lq'].initializer, inputs['hq'].initializer])

            def identity_function(input):
                return input

            d_arch = CyclicGAN.generate_dis_arch(patch=[1, 64, 64, 3], training=False, norm_func=identity_function)
            with tf.variable_scope('Discriminator') as scope:
                real = CyclicGAN.build_arch(inputs['lq'].get_next(), d_arch, 'LQ_Dis', graph)[-1]
                scope.reuse_variables()
                fake = CyclicGAN.build_arch(inputs['hq'].get_next(), d_arch, 'LQ_Dis', graph)[-1]

            opt = tf.train.AdamOptimizer(0.002, beta1=0.5)
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.square(fake)) + tf.reduce_mean(tf.squared_difference(1.0, real))
            with tf.name_scope('train_op'):
                var_list = [v for v in tf.trainable_variables() if 'LQ_Dis' in str(v)]
                top = opt.minimize(loss, var_list=var_list)
            sess.run(tf.global_variables_initializer())
            for i in range(steps):
                dfake, dreal, closs, _ = sess.run([fake, real, loss, top])
                print('step {}:\n\tfake {}\n\treal {}\n\tloss {}'.format(i, dfake, dreal, closs))

        print('duration: {}'.format(round(time.time() - start, 3)))


if __name__ == '__main__':

    main_test()
    tf.nn.relu6
