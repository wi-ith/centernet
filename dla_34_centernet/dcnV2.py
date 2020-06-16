import tensorflow as tf
import tensorflow.contrib as tc


def DeformableConv2D(input,
                     output_dims,
                     kernel_size,
                     stride,
                     idx,
                     seperable=False,
                     activation_fn=None,
                     normalizer_fn=None,
                     normalizer_params=None,
                     biases_initializer=None
                     ):
    with tf.variable_scope('DeformableConv2D'+'_'+idx, reuse=tf.AUTO_REUSE):

        if seperable:
            offsets = tc.layers.separable_conv2d(input, None, 3, 1,
                                                 stride=1,
                                                 activation_fn=tf.nn.relu6,
                                                 normalizer_fn=None, normalizer_params=None,
                                                 rate=1,
                                                 scope='offset_conv_depthwise')

            offsets = tc.layers.conv2d(offsets, kernel_size * kernel_size * 2, 1, activation_fn=tf.nn.sigmoid,
                                       normalizer_fn=None, normalizer_params=None,
                                       scope='offset_conv_pointwise')

            modulation = tc.layers.separable_conv2d(input, None, 3, 1,
                                                    stride=1,
                                                    activation_fn=tf.nn.relu6,
                                                    normalizer_fn=None, normalizer_params=None,
                                                    rate=1,
                                                    scope='weights_conv_depthwise')

            modulation = tc.layers.conv2d(modulation, kernel_size * kernel_size, 1, activation_fn=tf.nn.sigmoid,
                                          normalizer_fn=None, normalizer_params=None,
                                          scope='weights_conv_pointwise')

            modulation = tf.sigmoid(modulation)

        else:
            offsets = tc.layers.conv2d(input,
                                       kernel_size*kernel_size*2,
                                       kernel_size,
                                       stride=stride,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope='offset_conv')

            modulation = tc.layers.conv2d(input,
                                       kernel_size*kernel_size,
                                       kernel_size,
                                       stride=stride,
                                       activation_fn=tf.nn.sigmoid,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope='weights_conv')
            modulation = tf.sigmoid(modulation)



        input_shape = tf.shape(input)

        # [b, h, w, 2c] -> [b*c, h, w, 2]
        offsets = tf.transpose(offsets, [0,3,1,2])
        offsets = tf.reshape(offsets, [offsets.shape[0], -1, 2, offsets.shape[2],  offsets.shape[3]])
        offsets = tf.transpose(offsets, [0, 1, 3, 4, 2])
        offsets = tf.reshape(offsets, [-1, offsets.shape[2], offsets.shape[3], 2])

        # [b, h, w, c] -> [b*c, h, w]
        modulation = tf.transpose(modulation, [0,3,1,2])
        modulation = tf.reshape(modulation,[-1, modulation.shape[2],  modulation.shape[3]])

        grid_t = tf.reshape(tf.range(input_shape[1]),[-1,1])
        grid_l = tf.reshape(tf.range(input_shape[2]),[1,-1])
        grid_b = tf.reshape(tf.range(input_shape[0]),[-1,1,1,1])
        grid_bk = tf.tile(grid_b,[kernel_size*kernel_size,1,1,1])

        grid_t = tf.tile(grid_t, [1, input_shape[2]])
        grid_l = tf.tile(grid_l, [input_shape[1], 1])
        grid_tl = tf.stack([grid_t,grid_l],axis=-1)
        grid_tl = tf.tile(tf.expand_dims(grid_tl, axis=0), [offsets.shape[0], 1, 1, 1])

        coordi_yx = tf.cast(grid_tl, dtype=tf.float32) + offsets
        coordi_y_clip = tf.clip_by_value(coordi_yx[...,0],
                                         clip_value_min=0.,
                                         clip_value_max=tf.cast(input_shape[1], dtype=tf.float32) - 1.)
        coordi_x_clip = tf.clip_by_value(coordi_yx[..., 1],
                                         clip_value_min=0.,
                                         clip_value_max=tf.cast(input_shape[2], dtype=tf.float32) - 1.)
        coordi_yx = tf.stack([coordi_y_clip,coordi_x_clip], axis=-1)
        grid_bk = tf.cast(tf.tile(grid_bk, [1, input_shape[1], input_shape[2],1]),dtype=tf.float32)
        coordi_bkyx = tf.concat([grid_bk, coordi_yx],axis=-1)

        coordi_bktl = tf.cast(tf.floor(coordi_bkyx), dtype=tf.int32)
        coordi_bkbr = coordi_bktl + tf.constant(1)
        coordi_bktr = tf.stack([coordi_bktl[...,0], coordi_bktl[...,1], coordi_bkbr[...,2]],axis=-1)
        coordi_bkbl = tf.stack([coordi_bktl[...,0], coordi_bktl[...,2], coordi_bkbr[...,1]],axis=-1)

        #[batch, kernel*kernel, top, left, channel]
        var_bktlc = tf.gather_nd(input,tf.reshape(coordi_bktl,[-1,3]))
        var_tl = tf.reshape(var_bktlc,[input_shape[0]*kernel_size*kernel_size,
                                          input_shape[1],
                                          input_shape[2],
                                          input_shape[3]])

        # [batch, kernel*kernel, top, right, channel]
        var_bktrc = tf.gather_nd(input, tf.reshape(coordi_bktr, [-1, 3]))
        var_tr = tf.reshape(var_bktrc, [input_shape[0] * kernel_size * kernel_size,
                                           input_shape[1],
                                           input_shape[2],
                                           input_shape[3]])

        # [batch, kernel*kernel, bottom, left, channel]
        var_bkblc = tf.gather_nd(input, tf.reshape(coordi_bkbl, [-1, 3]))
        var_bl = tf.reshape(var_bkblc, [input_shape[0] * kernel_size * kernel_size,
                                           input_shape[1],
                                           input_shape[2],
                                           input_shape[3]])

        #[batch, kernel*kernel, bottom, right, channel]
        var_bkbrc = tf.gather_nd(input, tf.reshape(coordi_bkbr, [-1, 3]))
        var_br = tf.reshape(var_bkbrc, [input_shape[0] * kernel_size * kernel_size,
                                           input_shape[1],
                                           input_shape[2],
                                           input_shape[3]])

        offset_y = tf.expand_dims(coordi_yx[...,0] - tf.floor(coordi_yx[...,0]),axis=-1)
        offset_x = tf.expand_dims(coordi_yx[...,1] - tf.floor(coordi_yx[...,1]),axis=-1)

        offset_y = tf.tile(offset_y,[1,1,1,input_shape[3]])
        offset_x = tf.tile(offset_x,[1,1,1,input_shape[3]])

        var_top = var_tl + (var_tr - var_tl)*offset_x
        var_bottom = var_bl + (var_br - var_bl)*offset_x
        var_final = var_top + (var_bottom - var_top)*offset_y

        modulation = tf.expand_dims(modulation,axis = -1)
        var_final = var_final * modulation

        var_final = tf.reshape(var_final,[input_shape[0],kernel_size,kernel_size,input_shape[1],input_shape[2],input_shape[3]])
        var_final = tf.transpose(var_final,[0,1,3,2,4,5])
        var_final = tf.reshape(var_final, [input_shape[0],kernel_size*input_shape[1],kernel_size,input_shape[2],input_shape[3]])
        var_final = tf.reshape(var_final, [input_shape[0],kernel_size*input_shape[1],kernel_size*input_shape[2],input_shape[3]])

        if seperable:
            var_final = tc.layers.separable_conv2d(var_final, None, kernel_size, 1,
                                                   stride=kernel_size,
                                                   activation_fn=tf.nn.relu6,
                                                   normalizer_fn=normalizer_fn,
                                                   normalizer_params=normalizer_params,
                                                   padding='VALID',
                                                   rate=1,
                                                   scope='output_conv_depthwise')

            var_final = tc.layers.conv2d(var_final, output_dims, 1,
                                         activation_fn=tf.nn.relu6,
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_params,
                                         scope='output_conv_pointwise')

        else:
            var_final = tc.layers.conv2d(var_final,
                                         output_dims,
                                         kernel_size,
                                         stride=kernel_size,
                                         padding='VALID',
                                         activation_fn=activation_fn,
                                         normalizer_fn=normalizer_fn,
                                         biases_initializer=biases_initializer,
                                         normalizer_params=normalizer_params,
                                         scope='output_conv')

        return var_final
