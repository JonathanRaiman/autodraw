import tensorflow as tf
import numpy as np

from model import Model
import ops
from skimage import draw

def build_model(figsize, filters, kernel_size, hidden_sizes, lstm_size, lr,
                max_steps, fixed_sprites, num_sprites, decoder="rnn"):
    x, y, channels = figsize
    global_step = tf.train.create_global_step()
    with tf.variable_scope("Inputs"):
        inputs = tf.placeholder(shape=[None] + list(figsize),
          name="target_image", dtype=tf.uint8)
        inputs = tf.cast(inputs, tf.float32) / 255.0
        is_training = tf.placeholder_with_default(False,
                                                  name="is_training",
                                                  shape=[])
        max_steps = tf.placeholder_with_default(max_steps, name="max_steps",
                                                shape=[])
        if fixed_sprites:
          if num_sprites < 1:
              xx, yy = draw.circle(x * 0.5, x * 0.5, radius=x/2, shape=(x, y))
              canvas = np.zeros((1, 1, x, y, 1), dtype=np.float32)
              canvas[:, xx, yy, :] = 1.0
              sprites = tf.get_variable("sprites",
                                        initializer=canvas,
                                        trainable=False,
                                        dtype=tf.float32)
          else:
              sprites = tf.get_variable("sprites",
                                        shape=[1, num_sprites, x, y, 1],
                                        initializer=tf.random_uniform_initializer(0.0, 1.0),
                                        trainable=True,
                                        dtype=tf.float32)
          sprites = tf.tile(sprites, [tf.shape(inputs)[0], 1, 1, 1, 1])
        else:
          sprites = None

    with tf.variable_scope("Encoder"):
        out = tf.contrib.layers.flatten(
            tf.contrib.layers.conv2d(inputs, num_outputs=filters,
                                       scope="Conv2D", kernel_size=kernel_size))

        for idx, hsize in enumerate(hidden_sizes):
            conditioner = tf.contrib.layers.fully_connected(out,
                                                            num_outputs=hsize,
                                                            scope="FC%d" % (idx,))

    with tf.variable_scope("Decoder"):
        canvas = tf.zeros_like(inputs) + 1.0
        if decoder == "rnn":
            out = ops.rnn_decoder(canvas, conditioner, lstm_size,
                                  max_steps=max_steps,
                                  inputs=inputs,
                                  sprites=sprites,
                                  fixed_sprites=fixed_sprites,
                                  global_step=global_step,
                                  is_training=is_training)
        elif decoder == "mlp":
            out = ops.mlp_decoder(canvas, conditioner, lstm_size,
                                  max_steps=max_steps,
                                  sprites=sprites,
                                  fixed_sprites=fixed_sprites,
                                  global_step=global_step,
                                  is_training=is_training)
        elif decoder == "dummy":
            out = tf.contrib.layers.fully_connected(
                conditioner,
                num_outputs=x * y  * channels,
                activation_fn=tf.nn.sigmoid)
            out = tf.reshape(out, [tf.shape(out)[0], x, y, channels])
        else:
            raise ValueError("wrong decoder")

    with tf.variable_scope("Loss"):
        loss = tf.reduce_mean(tf.abs(out - inputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gradients = optimizer.compute_gradients(loss)
        grads, gnorm = tf.clip_by_global_norm([grad for grad, _ in gradients], clip_norm=0.1)
        tvars = tf.trainable_variables()
        grads = [grad + tf.random_normal(shape=tf.shape(grad), stddev=1e-8)
                 for grad in grads]

        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    out_uint8 = tf.cast(out * 255.0, tf.uint8)
    return Model([inputs], [out_uint8], loss, train_op, figsize=figsize,
                 training=is_training, grad_norm=gnorm, max_steps=max_steps,
                 global_step=global_step)

