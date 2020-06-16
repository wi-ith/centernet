# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
import tensorflow.contrib as tc
import dcnV2 as dcn
from tensorflow.contrib.framework.python.ops import arg_scope

_WEIGHT_DECAY = 1e-4

FLAGS = tf.app.flags.FLAGS

class DLA_34(object):
    def __init__(self, is_training=True, input_size=224):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training,
                          'scale': True,
                          'center': True,
                          'decay': 0.9997,
                          'epsilon': 0.001,
                          }
        self.use_transpose=False

    def basic_block(self, input, output_dims, stride=1, scope=None, dilation=1):
        with tf.variable_scope(scope+'_basic_block', reuse=tf.AUTO_REUSE):
            residual = tf.identity(input)
            input_dims = tf.shape(input)[0]

            if stride > 1:
                residual = tf.identity(tf.nn.max_pool(residual,
                                                      ksize=(1,2,2,1),
                                                      strides=(1,2,2,1),
                                                      padding='SAME'))

            if input_dims!=output_dims:
                residual = tc.layers.conv2d(residual, output_dims, 1,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='project')



            output = tc.layers.conv2d(input, output_dims, 3,
                                      stride=stride,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_1')
            output = tf.nn.relu(output)
            output = tc.layers.conv2d(output, output_dims, 3,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_2')
            output = output + residual
            output = tf.nn.relu(output)
            return output, residual


    def root_block(self, *input, scope=None, output_dims, residual=False):
        with tf.variable_scope(scope + '_root_block', reuse=tf.AUTO_REUSE):
            output = tf.concat(input, axis=3)
            output = tc.layers.conv2d(output, output_dims, 1,
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=self.normalizer,
                                      normalizer_params=self.bn_params,
                                      scope='conv_1')
            if residual:
                output = output + input[0]
            output = tf.nn.relu(output)
            return output

    def up_layer(self,input_, output_dims, resize_shape, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.use_transpose:
                output = tc.layers.conv2d_transpose(input_,
                                                    output_dims,
                                                    4,
                                                    stride=2,
                                                    padding='SAME',
                                                    biases_initializer=None)
            else:
                output = tc.layers.separable_conv2d(input_, None, 3, 1,
                                                    stride=1,
                                                    activation_fn=None,
                                                    normalizer_fn=None, normalizer_params=None,
                                                    rate=1)
                output = tf.image.resize_bilinear(output,
                                                  resize_shape,
                                                  align_corners=True)
            return output


    def _build_model(self, image):
        self.i = 0
        with arg_scope([tc.layers.conv2d],
                       weights_regularizer=tc.layers.l2_regularizer(_WEIGHT_DECAY)):
            with tf.variable_scope('FeatureExtractor/MobilenetV2', reuse=tf.AUTO_REUSE):
                # image_copy=tf.identity(image)
                #base_layer
                output = tc.layers.conv2d(image, 16, 7, 1,
                                          activation_fn=tf.nn.relu,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params,
                                          scope='base_layer')  # base layer
                #level 0
                output = tc.layers.conv2d(output, 16, 3, 1,
                                          activation_fn=tf.nn.relu,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params,
                                          scope='level_0')  # level 0
                #level 1
                output = tc.layers.conv2d(output, 32, 3, 2,
                                          activation_fn=tf.nn.relu,
                                          normalizer_fn=self.normalizer,
                                          normalizer_params=self.bn_params,
                                          scope='level_1')  # level 1

                #level 2
                with tf.variable_scope('level_2', reuse=tf.AUTO_REUSE):
                    output_lv2_1, _ = self.basic_block(output, 64, stride=2, scope='output_lv2_1', dilation=1)
                    output_lv2_2, _ = self.basic_block(output_lv2_1, 64, stride=1, scope='output_lv2_2', dilation=1)
                    output_lv2_root = self.root_block(output_lv2_1,
                                                      output_lv2_2,
                                                      scope='output_lv2_root',
                                                      output_dims=64)

                #level 3
                with tf.variable_scope('level_3', reuse=tf.AUTO_REUSE):
                    output_lv3_1_1, output_lv3_1_1_residual = self.basic_block(output_lv2_root, 128, stride=2, scope='output_lv3_1_1', dilation=1)
                    output_lv3_1_2, _ = self.basic_block(output_lv3_1_1, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                    output_lv3_1_root = self.root_block(output_lv3_1_1,
                                                        output_lv3_1_2,
                                                        scope='output_lv3_1_root',
                                                        output_dims=128)

                    output_lv3_2_1, _ = self.basic_block(output_lv3_1_root, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                    output_lv3_2_2, _ = self.basic_block(output_lv3_2_1, 128, stride=1, scope='output_lv3_1_2', dilation=1)
                    output_lv3_2_root = self.root_block(output_lv3_2_2,
                                                        output_lv3_2_1,
                                                        output_lv3_1_1_residual,
                                                        output_lv3_1_root,
                                                        scope='output_lv3_1_2',
                                                        output_dims=128)

                #level 4
                with tf.variable_scope('level_4', reuse=tf.AUTO_REUSE):
                    output_lv4_1_1, output_lv4_1_1_residual = self.basic_block(output_lv3_2_root, 256, stride=2, scope='output_lv4_1_1', dilation=1)
                    output_lv4_1_2, _ = self.basic_block(output_lv4_1_1, 256, stride=1, scope='output_lv4_1_2', dilation=1)
                    output_lv4_1_root = self.root_block(output_lv4_1_1,
                                                        output_lv4_1_2,
                                                        scope='output_lv4_1_root',
                                                        output_dims=256)

                    output_lv4_2_1, _ = self.basic_block(output_lv4_1_root, 256, stride=1, scope='output_lv4_2_1', dilation=1)
                    output_lv4_2_2, _ = self.basic_block(output_lv4_2_1, 256, stride=1, scope='output_lv4_2_2', dilation=1)
                    output_lv4_2_root = self.root_block(output_lv4_2_2,
                                                        output_lv4_2_1,
                                                        output_lv4_1_1_residual,
                                                        output_lv4_1_root,
                                                        scope='output_lv4_2_root',
                                                        output_dims=256)

                #level 5
                with tf.variable_scope('level_5', reuse=tf.AUTO_REUSE):
                    output_lv5_1, output_lv5_1_residual = self.basic_block(output_lv4_2_root, 512, stride=2, scope='output_lv5_1', dilation=1)
                    output_lv5_2, _ = self.basic_block(output_lv5_1, 512, stride=1, scope='output_lv5_2', dilation=1)
                    output_lv5_root = self.root_block(output_lv5_1,
                                                      output_lv5_2,
                                                      output_lv5_1_residual,
                                                      scope='output_lv5_root',
                                                      output_dims=512)

                #IDA_0 (16x16)=>(32x32) output stride -> 16
                ida0_proj1 = dcn.DeformableConv2D(output_lv5_root,
                                                  output_dims=256,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida0_proj1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                ida0_up1 = self.up_layer(ida0_proj1, 256, tf.shape(output_lv4_2_root)[1:3], 'ida0_up1')

                ida0_proj2 = dcn.DeformableConv2D(output_lv4_2_root,
                                                  output_dims=256,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida0_proj2',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                ida0_nodesum1 = tf.identity(ida0_up1 + ida0_proj2)

                ida0_node1 = dcn.DeformableConv2D(ida0_nodesum1,
                                                  output_dims=256,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida0_node1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                #IDA_1 (32x32)=>(64x64) output stride -> 8
                ida1_proj1 = dcn.DeformableConv2D(output_lv4_2_root,
                                                  output_dims=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida1_proj1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                ida1_up1 = self.up_layer(ida1_proj1, 128, tf.shape(output_lv3_2_root)[1:3], 'ida1_up1')

                ida1_proj2 = dcn.DeformableConv2D(output_lv3_2_root,
                                                  output_dims=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida1_proj2',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                ida1_nodesum1 = tf.identity(ida1_up1 + ida1_proj2)

                ida1_node1 = dcn.DeformableConv2D(ida1_nodesum1,
                                                  output_dims=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida1_node1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida1_proj3 = dcn.DeformableConv2D(ida0_node1,
                                                  output_dims=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida1_node1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida1_up2 = self.up_layer(ida1_proj3, 128, tf.shape(output_lv3_2_root)[1:3], 'ida1_up2')

                ida1_nodesum2 = tf.identity(ida1_up2 + ida1_node1)

                ida1_node2 = dcn.DeformableConv2D(ida1_nodesum2,
                                                  output_dims=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida1_node2',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                #DLA_2 (64x64)=>(128x128) output stride -> 4
                ida2_proj1 = dcn.DeformableConv2D(output_lv3_2_root,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_proj1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )
                ida2_up1 = self.up_layer(ida2_proj1, 64, tf.shape(output_lv2_root)[1:3], 'ida2_up1')

                ida2_proj2 = dcn.DeformableConv2D(output_lv2_root,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_proj2',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None
                                                  )

                ida2_nodesum1 = tf.identity(ida2_up1 + ida2_proj2)

                ida2_node1 = dcn.DeformableConv2D(ida2_nodesum1,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_node1',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida2_proj3 = dcn.DeformableConv2D(ida1_node1,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_proj3',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida2_up2 = self.up_layer(ida2_proj3, 64, tf.shape(output_lv2_root)[1:3], 'ida2_up2')

                ida2_nodesum2 = tf.identity(ida2_up2 + ida2_node1)

                ida2_node2 = dcn.DeformableConv2D(ida2_nodesum2,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_node2',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida2_proj4 = dcn.DeformableConv2D(ida1_node2,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_proj4',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                ida2_up3 = self.up_layer(ida2_proj4, 64, tf.shape(output_lv2_root)[1:3], 'ida2_up3')

                ida2_nodesum3 = tf.identity(ida2_up3 + ida2_node2)

                ida2_node3 = dcn.DeformableConv2D(ida2_nodesum3,
                                                  output_dims=64,
                                                  kernel_size=3,
                                                  stride=1,
                                                  idx='ida2_node3',
                                                  seperable=False,
                                                  activation_fn=None,
                                                  normalizer_fn=self.normalizer,
                                                  normalizer_params=self.bn_params,
                                                  biases_initializer=None)

                #ida sum
                with tf.variable_scope('sum1', reuse=tf.AUTO_REUSE):
                    sum1_proj = dcn.DeformableConv2D(ida1_node2,
                                                      output_dims=64,
                                                      kernel_size=3,
                                                      stride=1,
                                                      idx='sum1_proj',
                                                      seperable=False,
                                                      activation_fn=None,
                                                      normalizer_fn=self.normalizer,
                                                      normalizer_params=self.bn_params,
                                                      biases_initializer=None)

                    sum1_up = self.up_layer(sum1_proj, 64, tf.shape(output_lv2_root)[1:3], 'sum1_up')

                    sum1_nodesum = tf.identity(sum1_up + ida2_node3)


                    sum1_node = dcn.DeformableConv2D(sum1_nodesum,
                                                         output_dims=64,
                                                         kernel_size=3,
                                                         stride=1,
                                                         idx='sum1_node',
                                                         seperable=False,
                                                         activation_fn=None,
                                                         normalizer_fn=self.normalizer,
                                                         normalizer_params=self.bn_params,
                                                         biases_initializer=None)

                with tf.variable_scope('sum2', reuse=tf.AUTO_REUSE):
                    sum2_proj = dcn.DeformableConv2D(ida0_node1,
                                                     output_dims=64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     idx='sum2_proj',
                                                     seperable=False,
                                                     activation_fn=None,
                                                     normalizer_fn=self.normalizer,
                                                     normalizer_params=self.bn_params,
                                                     biases_initializer=None)

                    sum2_up = self.up_layer(sum2_proj, 64, tf.shape(output_lv2_root)[1:3], 'sum2_up')

                    sum2_nodesum = tf.identity(sum2_up + sum1_node)

                    sum2_node = dcn.DeformableConv2D(sum2_nodesum,
                                                     output_dims=64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     idx='sum2_node',
                                                     seperable=False,
                                                     activation_fn=None,
                                                     normalizer_fn=self.normalizer,
                                                     normalizer_params=self.bn_params,
                                                     biases_initializer=None)

                with tf.variable_scope('sum3', reuse=tf.AUTO_REUSE):
                    sum3_proj = dcn.DeformableConv2D(output_lv5_root,
                                                 output_dims=64,
                                                 kernel_size=3,
                                                 stride=1,
                                                 idx='sum2_proj',
                                                 seperable=False,
                                                 activation_fn=None,
                                                 normalizer_fn=self.normalizer,
                                                 normalizer_params=self.bn_params,
                                                 biases_initializer=None)

                    sum3_up = self.up_layer(sum3_proj, 64, tf.shape(output_lv2_root)[1:3], 'sum3_up')

                    sum3_nodesum = tf.identity(sum3_up + sum2_node)

                    sum3_node = dcn.DeformableConv2D(sum3_nodesum,
                                                     output_dims=64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     idx='sum3_node',
                                                     seperable=False,
                                                     activation_fn=None,
                                                     normalizer_fn=self.normalizer,
                                                     normalizer_params=self.bn_params,
                                                     biases_initializer=None)

                return sum3_node



'''
A  |  <ida2> a node(proj2(A) + (proj1(B,64)-up)) | b node(ida2-a + proj3(ida1-a,64)-up) | c node(ida2-b + proj5(ida1-b,64)-up)

B  |  <ida1> a node(proj2(B) + (proj1(C,128)-up)) | b node(ida1-a + proj3(ida0-a,128)-up)   -> sum1<proj(proj(ida1-b,64)-up + ida2-c)>

C  |  <ida0> a node(proj2(C) + (proj1(D,256)->up))                                          -> sum2<proj(proj(ida0 a,64)-up + sum1)>

D  |                                                                                        -> sum3<proj(proj(D,64)-up + sum2)>
'''








