import models
import tensorflow as tf
import os
import argparse
tf.logging.set_verbosity(tf.logging.ERROR)


def parse(serialized):

    features = {'image': tf.FixedLenFeature([], tf.string)}
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    image_raw = parsed_example['image']
    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, (512,512,1))

    return image


def get_files(path, quality):

    all_files = os.listdir(path)
    relevant_files = [os.path.join(path, f) for f in all_files if quality in f]

    return relevant_files


def preprocess(image):

    global MIN, MAX

    if MIN == 'mean':

        indices = tf.greater(image, 0)
        mean, std = tf.nn.moments(tf.boolean_mask(image, indices), axes=0)
        vmin = mean - 0.5*std
    else:
        vmin = int(MIN)

    vmax = int(MAX)

    lower = tf.cast(tf.less(vmin * tf.ones_like(image), image), image.dtype)
    upper = tf.cast(tf.greater(vmax * tf.ones_like(image), image), image.dtype)

    return image * upper * lower


def build_dataset(filedir, quality, buffer_size=128, batch_size=16):
    files = get_files(filedir, quality)
    dataset = tf.data.TFRecordDataset(filenames=files, num_parallel_reads=7)
    dataset = dataset.map(parse).map(preprocess)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(1)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(batch_size)

    return dataset


def parse_functions(dictionary):
    global func_mappings
    items = dictionary.items()
    parsed = {'gen': {}, 'dis': {}}
    for i in items:
        key1 = 'gen' if 'gen' in i[0] else 'dis'
        key2 = 'activation' if 'activation' in i[0] else 'norm_func'
        parsed[key1][key2] = func_mappings[i[1]]

    return parsed


def construct_logdir(root, args):
    dir = os.path.join(root,
                       'gen_norm-{}_act-{}_scale-{}_res-{}x{}_f-{}'.format(args['gen_norm'], args['gen_activation'],
                                                                         args['scale_factor'], args['n_res_blocks'],
                                                                         args['n_res_layers'], args['generator_filters']),
                       'dis_norm-{}_act-{}_f-{}'.format(args['dis_norm'], args['dis_activation'],
                                                        args['discriminator_filters']),
                       '{}-cyc_{}-dis'.format(args['loss_weights'][0], args['loss_weights'][1]), '')
    return dir


func_mappings = {
    'relu': tf.nn.relu,
    'lrelu': tf.nn.leaky_relu,
    'selu': tf.nn.selu,
    'id': models.identity,
    'bn': tf.layers.batch_normalization,
    'in': models.instance_norm
}


parser = argparse.ArgumentParser()
parser.add_argument('-ga', '--generator-activation', type=str, dest='gen_activation', default='relu')
parser.add_argument('-da', '--discriminator-activation', type=str, dest='dis_activation', default='lrelu')
parser.add_argument('-gn', '--generator-normalization', type=str, dest='gen_norm', default='in')
parser.add_argument('-dn', '--discriminator-normalization', type=str, dest='dis_norm', default='in')
parser.add_argument('-l', '--loss-weights', nargs=2, type=int, dest='loss_weights', default=[1, 1])
parser.add_argument('-sf', '--scaling-factor', type=int, dest='scale_factor', default=3)
parser.add_argument('-nrb', '--n-res-blocks', type=int, dest='n_res_blocks', default=6)
parser.add_argument('-gf', '--generator-filters', type=int, dest='generator_filters', default=16)
parser.add_argument('-df', '--discriminator-filters', type=int, dest='discriminator_filters', default=1)
parser.add_argument('-nrl', '--n-res-layers', type=int, dest='n_res_layers', default=3)
parser.add_argument('-s', '--skip', dest='skip_connections', default=False, const=True, action='store_const')
parser.add_argument('-n', '--name', type=str, dest='name', default='cycoct')
parser.add_argument('-c', '--checkpoint', type=str, dest='checkpoint', default='none')
parser.add_argument('-lr', '--learning-rate', type=float, dest='learning-rate', default=0.0002)
parser.add_argument('-e', '--epochs', nargs=2, type=int, dest='epochs', default=[200, 100])
parser.add_argument('-b', '--batch-size', type=int, dest='batch-size', default=2)
parser.add_argument('-min', '--min', type=str, dest='min', default='0')
parser.add_argument('-max', '--max', type=str, dest='max', default='1')
args = vars(parser.parse_args())

# settings
#filedir = os.path.join('/media/network/ImageRepository', 'imanakov', 'OCT_Quality_TFRecords', '')
filedir = os.path.join('/aug-learning/Datasets/', 'HQ_LQ_OCTs', 'OCT_Quality_TFRecords2', '')
checkpoint = args.pop('checkpoint')
learning_rate = args.pop('learning-rate')
epochs = args.pop('epochs')
batch_size = args.pop('batch-size')
logdir = '/outdir/{}'.format(args.pop('name'))


logdir = construct_logdir(logdir, args)
lw = args.pop('loss_weights')
scale_factor = args.pop('scale_factor')
n_residuals = args.pop('n_res_blocks')
n_reslayers = args.pop('n_res_layers')
dis_filters = args.pop('discriminator_filters')
gen_filters = args.pop('generator_filters')
MIN = args.pop('min')
MAX = args.pop('max')
skip = args.pop('skip_connections')
input_size = 512
funcs = parse_functions(args)

graph = tf.Graph()
tf.logging.set_verbosity(tf.logging.WARN)
with graph.as_default():
    # generate datasets
    print('parsing data...')
    low_dataset = build_dataset(filedir, 'low', batch_size=batch_size)
    high_dataset = build_dataset(filedir, 'high', batch_size=batch_size)

    # build GAN model
    print('building model...')
    low_it = low_dataset.make_initializable_iterator()
    high_it = high_dataset.make_initializable_iterator()
    isTraining = True
    g_arch = models.CyclicGAN.generate_gen_arch(scale_factor, n_residuals, n_reslayers, training=isTraining, filters=gen_filters, skip_conn=skip, **funcs['gen'])
    d_arch = models.CyclicGAN.generate_dis_arch(input_size, training=isTraining, filters=dis_filters, **funcs['dis'])
    inputs = {'lq': low_it, 'hq': high_it}

    cgan = models.CyclicGAN(g_arch, d_arch, inputs, isTraining, logdir, graph=graph, loss_weights=lw, n_saves=100, pool_size=32, interval=25, learning_rate=learning_rate)

    if checkpoint is not 'none':
        cgan.load(logdir + 'cGAN.ckpt-' + checkpoint)
        assign = cgan.global_step.assign(int(checkpoint))
        cgan.sess.run(assign)

# train the GAN for 200 epochs
print('training model...')
cgan.train(*epochs)

