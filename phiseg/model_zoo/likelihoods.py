import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def _batch_mv(bmat, bvec):
    bmat = tf.tile(tf.expand_dims(bmat, axis=0), (bvec.shape[0].value,) + (1,) * len(bmat.shape))
    bvec = tf.expand_dims(bvec, axis=-1)
    return tf.squeeze(tf.matmul(bmat, bvec), axis=-1)


# port from pytorch, fast and dirty no checks, not meant to ever be reused
class LowRankMultivariateNormal(object):
    def __init__(self, loc, cov_factor, cov_diag):
        self.loc = loc
        self.cov_diag = cov_diag
        self.cov_factor = cov_factor
        self.batch_shape = tuple(tf.unstack(tf.shape(self.loc)[:-1]))
        self.event_shape = tuple(tf.unstack(tf.shape(self.loc)[-1:]))

    @staticmethod
    def get_shape(shape):
        return tuple(s.value if s.value is not None else -1 for s in shape)

    def rsample(self, sample_shape):
        assert isinstance(sample_shape, tuple)
        shape = sample_shape + self.batch_shape + self.event_shape
        W_shape = shape[:-1] + (self.cov_factor.shape[-1].value,)
        eps_W = tf.random.normal(W_shape, dtype=self.loc.dtype)
        eps_D = tf.random.normal(shape, dtype=self.loc.dtype)
        return (self.loc + _batch_mv(self.cov_factor, eps_W) + tf.sqrt(self.cov_diag) * eps_D)


def proposed(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    x = kwargs.get('x')

    resolution_levels = kwargs.get('resolution_levels', 7)
    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

    conv_unit = layers.conv2D
    deconv_unit = lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2)

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.averagepool2D(enc[ii - 1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_3' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels - 1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj - 1][-1]

                dec[jj].append(deconv_unit(next_inp))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii - 1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm,
                                         add_bias=add_bias))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_3' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        net = dec[-1][-1]

        recomb = conv_unit(net, 'recomb_0', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)

        rank = kwargs.get('rank', 10)
        epsilon = kwargs.get('epsilon', 1e-5)
        mean = layers.conv2D(recomb, 'mean', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
        log_cov_diag = layers.conv2D(recomb, 'diag', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)
        cov_factor = layers.conv2D(recomb, 'factor', num_filters=n_classes * rank, kernel_size=(1, 1),
                                   activation=tf.identity)

        shape = image_size[:-1] + (n_classes,)
        flat_size = np.prod(shape)
        mean = tf.reshape(mean, [-1, flat_size])
        cov_diag = tf.exp(tf.reshape(log_cov_diag, [-1, flat_size])) + epsilon
        cov_factor = tf.reshape(cov_factor, [-1, flat_size, rank])
        dist = LowRankMultivariateNormal(loc=mean, cov_diag=cov_diag, cov_factor=cov_factor)

        s = dist.rsample((1,))
        s = tf.reshape(s, (-1,) + shape)
        return [[s], dist]


def det_unet2D(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    x = kwargs.get('x')

    resolution_levels = kwargs.get('resolution_levels', 7)
    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

    conv_unit = layers.conv2D
    deconv_unit = lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2)

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.averagepool2D(enc[ii - 1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_3' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels - 1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj - 1][-1]

                dec[jj].append(deconv_unit(next_inp))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii - 1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm,
                                         add_bias=add_bias))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_3' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        net = dec[-1][-1]

        recomb = conv_unit(net, 'recomb_0', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)

        s = [layers.conv2D(recomb, 'prediction', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)]

        return s


def prob_unet2D(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    x = kwargs.get('x')

    z = z_list[0]

    resolution_levels = kwargs.get('resolution_levels', 7)
    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

    conv_unit = layers.conv2D
    deconv_unit = lambda inp: layers.bilinear_upsample2D(inp, 'upsample', 2)

    bs = tf.shape(x)[0]
    zdim = z.get_shape().as_list()[-1]

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.averagepool2D(enc[ii - 1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_3' % ii, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels - 1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj - 1][-1]

                dec[jj].append(deconv_unit(next_inp))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii - 1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm,
                                         add_bias=add_bias))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_3' % jj, num_filters=num_channels[ii], training=training,
                                         normalisation=norm, add_bias=add_bias))

        z_t = tf.reshape(z, tf.stack((bs, 1, 1, zdim)))

        broadcast_z = tf.tile(z_t, (1, image_size[0], image_size[1], 1))

        net = tf.concat([dec[-1][-1], broadcast_z], axis=-1)

        recomb = conv_unit(net, 'recomb_0', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_1', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)
        recomb = conv_unit(recomb, 'recomb_2', num_filters=num_channels[0], kernel_size=(1, 1), training=training,
                           normalisation=norm, add_bias=add_bias)

        s = [layers.conv2D(recomb, 'prediction', num_filters=n_classes, kernel_size=(1, 1), activation=tf.identity)]

        return s


def phiseg(z_list, training, image_size, n_classes, scope_reuse=False, norm=tfnorm.batch_norm, **kwargs):
    """
    This is a U-NET like arch with skips before and after latent space and a rather simple decoder
    """

    n0 = kwargs.get('n0', 32)
    num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

    def increase_resolution(x, times, num_filters, name):

        with tf.variable_scope(name):
            nett = x

            for i in range(times):
                nett = layers.bilinear_upsample2D(nett, 'ups_%d' % i, 2)
                nett = layers.conv2D(nett, 'z%d_post' % i, num_filters=num_filters, normalisation=norm,
                                     training=training)

        return nett

    with tf.variable_scope('likelihood') as scope:

        if scope_reuse:
            scope.reuse_variables()

        resolution_levels = kwargs.get('resolution_levels', 7)
        latent_levels = kwargs.get('latent_levels', 5)
        lvl_diff = resolution_levels - latent_levels

        post_z = [None] * latent_levels
        post_c = [None] * latent_levels

        s = [None] * latent_levels

        # Generate post_z
        for i in range(latent_levels):
            net = layers.conv2D(z_list[i], 'z%d_post_1' % i, num_filters=num_channels[i], normalisation=norm,
                                training=training)
            net = layers.conv2D(net, 'z%d_post_2' % i, num_filters=num_channels[i], normalisation=norm,
                                training=training)
            net = increase_resolution(net, resolution_levels - latent_levels, num_filters=num_channels[i],
                                      name='preups_%d' % i)

            post_z[i] = net

        # Upstream path
        post_c[latent_levels - 1] = post_z[latent_levels - 1]

        for i in reversed(range(latent_levels - 1)):
            ups_below = layers.bilinear_upsample2D(post_c[i + 1], name='post_z%d_ups' % (i + 1), factor=2)
            ups_below = layers.conv2D(ups_below, 'post_z%d_ups_c' % (i + 1), num_filters=num_channels[i],
                                      normalisation=norm, training=training)

            concat = tf.concat([post_z[i], ups_below], axis=3, name='concat_%d' % i)

            net = layers.conv2D(concat, 'post_c_%d_1' % i, num_filters=num_channels[i + lvl_diff], normalisation=norm,
                                training=training)
            net = layers.conv2D(net, 'post_c_%d_2' % i, num_filters=num_channels[i + lvl_diff], normalisation=norm,
                                training=training)

            post_c[i] = net

        # Outputs
        for i in range(latent_levels):
            s_in = layers.conv2D(post_c[i], 'y_lvl%d' % i, num_filters=n_classes, kernel_size=(1, 1),
                                 activation=tf.identity)
            s[i] = tf.image.resize_images(s_in, image_size[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return s
