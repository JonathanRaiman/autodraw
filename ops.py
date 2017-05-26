import numpy as np
import tensorflow as tf


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y,1,keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def gumbel_softmax_anneal(logits, min_val, max_val, anneal_rate, global_step):
    """"""
    tau = freeze_rate = min_val + tf.train.exponential_decay(
        max_val - min_val, global_step, 1000, anneal_rate, staircase=False)
    return gumbel_softmax(logits, tau)


def ThinPlateSpline(U, coord, vector, out_size):
    """Thin Plate Spline Spatial Transformer Layer
    TPS control points are arranged in arbitrary positions given by `coord`.
    U : float Tensor [num_batch, height, width, num_channels].
        Input Tensor.
    coord : float Tensor [num_batch, num_point, 2]
        Relative coordinate of the control points.
    vector : float Tensor [num_batch, num_point, 2]
        The vector on the control points.
    out_size: tuple of two integers [height, width]
        The size of the output of the network (height, width)
    ----------
    Reference :
        1. Spatial Transformer Network implemented by TensorFlow
            https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
        2. Thin Plate Spline Spatial Transformer Network with regular grids.
            https://github.com/iwyoo/TPS_STN-tensorflow
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output

    def _meshgrid(height, width, coord):
        x_t = tf.tile(
            tf.reshape(tf.linspace(-1.0, 1.0, width), [1, width]), [height, 1])
        y_t = tf.tile(
            tf.reshape(tf.linspace(-1.0, 1.0, height), [height, 1]), [1, width])

        x_t_flat = tf.reshape(x_t, (1, 1, -1))
        y_t_flat = tf.reshape(y_t, (1, 1, -1))

        num_batch = tf.shape(coord)[0]
        px = tf.expand_dims(coord[:,:,0], 2) # [bn, pn, 1]
        py = tf.expand_dims(coord[:,:,1], 2) # [bn, pn, 1]
        d2 = tf.square(x_t_flat - px) + tf.square(y_t_flat - py)
        r = d2 * tf.log(d2 + 1e-6) # [bn, pn, h*w]
        x_t_flat_g = tf.tile(x_t_flat, tf.stack([num_batch, 1, 1])) # [bn, 1, h*w]
        y_t_flat_g = tf.tile(y_t_flat, tf.stack([num_batch, 1, 1])) # [bn, 1, h*w]
        ones = tf.ones_like(x_t_flat_g) # [bn, 1, h*w]

        grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1) # [bn, 3+pn, h*w]
        return grid

    def _transform(T, coord, input_dim, out_size):
        num_batch = tf.shape(input_dim)[0]
        height = tf.shape(input_dim)[1]
        width = tf.shape(input_dim)[2]
        num_channels = tf.shape(input_dim)[3]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width, coord) # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = tf.matmul(T, grid) #
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    def _solve_system(coord, vector):
        num_batch  = tf.shape(coord)[0]
        num_point  = coord.get_shape()[1].value

        ones = tf.ones([num_batch, num_point, 1], dtype="float32")
        p = tf.concat([ones, coord], 2) # [bn, pn, 3]

        p_1 = tf.reshape(p, [num_batch, -1, 1, 3]) # [bn, pn, 1, 3]
        p_2 = tf.reshape(p, [num_batch, 1, -1, 3]) # [bn, 1, pn, 3]
        d2 = tf.reduce_sum(tf.square(p_1-p_2), 3) # [bn, pn, pn]
        r = d2 * tf.log(d2 + 1e-6) # [bn, pn, pn]

        zeros = tf.zeros([num_batch, 3, 3], dtype="float32")
        W_0 = tf.concat([p, r], 2) # [bn, pn, 3+pn]
        W_1 = tf.concat([
            zeros, tf.transpose(p, [0, 2, 1])], 2) # [bn, 3, pn+3]
        W = tf.concat([W_0, W_1], 1) # [bn, pn+3, pn+3]
        W_inv = tf.matrix_inverse(W)

        tp = tf.pad(coord+vector,
            [[0, 0], [0, 3], [0, 0]], "CONSTANT") # [bn, pn+3, 2]
        T = tf.matmul(W_inv, tp) # [bn, pn+3, 2]
        T = tf.transpose(T, [0, 2, 1]) # [bn, 2, pn+3]

        return T

    T = _solve_system(coord, vector)
    output = _transform(T, coord, U, out_size)
    return output


def paintbrush(inputs, sprite_size, sprites, is_training, global_step,
               fixed_sprites):
    # if there are choices for what sprite to use, create a
    # softmax over them:
    if fixed_sprites:
        if sprites.get_shape()[1].value != 1:
            with tf.variable_scope("PickSprite"):
                logits = tf.contrib.layers.fully_connected(inputs,
                                                           # num classes
                                                           num_outputs=sprites.get_shape()[1].value,
                                                           activation_fn=None)

                probs = gumbel_softmax_anneal(logits, min_val=0.5, max_val=5.0,
                    anneal_rate=0.99, global_step=global_step)
                weighted_sprites = sprites * tf.expand_dims(tf.expand_dims(tf.expand_dims(probs, 2), 2), 2)
                sprites = tf.reduce_sum(weighted_sprites, 1)
        else:
            sprites = sprites[:, 0]
    else:
        with tf.variable_scope("BuildSprite"):
            sprites = tf.contrib.layers.fully_connected(
                inputs, num_outputs=sprite_size[0] * sprite_size[1],
                activation_fn=tf.nn.sigmoid)
            sprites = tf.reshape(sprites, [tf.shape(sprites)[0],
                                           sprite_size[0],
                                           sprite_size[1],
                                           1])


    with tf.variable_scope("Spatial"):
        controls = tf.contrib.layers.fully_connected(inputs,
                                                     # x, y, size
                                                     num_outputs=16,
                                                     activation_fn=None)
        controls = tf.reshape(controls, [-1, 2, 4, 2])
        coord, vector = controls[:, 0, :], controls[:, 1, :]

        x, y = sprites.get_shape()[1].value, sprites.get_shape()[2].value
        tformed =  ThinPlateSpline(sprites, coord, vector, [x, y])

    with tf.variable_scope("Color"):
        # batch x colors
        colors = tf.contrib.layers.fully_connected(inputs, num_outputs=sprite_size[2],
                                                   activation_fn=tf.sigmoid)
    return tformed * tf.expand_dims(tf.expand_dims(colors, 1), 1)


def rnn_decoder(canvas, conditioner, lstm_size, max_steps, sprites,
                fixed_sprites, is_training, inputs, global_step):
    def cond(iteration, *args):
        return tf.less(iteration, max_steps)
    cell = tf.contrib.rnn.LSTMBlockCell(lstm_size)

    def body(iteration, state, old_canvas):
        new_hidden, new_state = cell(tf.concat([conditioner,
                                                tf.contrib.layers.flatten(old_canvas - inputs)], 1),
                                     state)
        new_canvas = old_canvas - paintbrush(new_hidden, sprites=sprites,
                                             sprite_size=[dim.value for dim in canvas.get_shape()[-3:]],
                                             fixed_sprites=fixed_sprites,
                                             is_training=is_training,
                                             global_step=global_step)
        return iteration + 1, new_state, new_canvas
    with tf.variable_scope("Conditioner"):
        initial_state = tf.contrib.layers.fully_connected(conditioner,
                                                          num_outputs=lstm_size,
                                                          activation_fn=None)
        initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state, initial_state)
    _, h, out = tf.while_loop(cond=cond,
                              body=body,
                              loop_vars=(0, initial_state, canvas))
    return out


def mlp_decoder(canvas, conditioner, hidden_size, max_steps, sprites,
                is_training, global_step, fixed_sprites):
    for i in range(max_steps):
        with tf.variable_scope("Step%d" % (i,)):
            h = tf.contrib.layers.fully_connected(
                tf.concat([conditioner, tf.contrib.layers.flatten(canvas)], 1),
                num_outputs=hidden_size,
                scope="Decode%d" % (i,)
            )
            brush = paintbrush(h, sprites=sprites,
                               sprite_size=[dim.value for dim in canvas.get_shape()[-3:]],
                               fixed_sprites=fixed_sprites,
                               is_training=is_training,
                               global_step=global_step)
            canvas -= brush
    return canvas


