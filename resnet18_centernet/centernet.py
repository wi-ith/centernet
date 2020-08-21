# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""


import tensorflow as tf

import model

import tensorflow.contrib.slim as slim

import tensorflow.contrib as tc


FLAGS = tf.app.flags.FLAGS

output_stride=4

def classifier(logits):
    with tf.variable_scope('hm_layer', reuse=tf.AUTO_REUSE):
        hm = tc.layers.conv2d(logits,
                              256,
                              3,
                              stride=1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=None,
                              normalizer_params=None)

        hm = tc.layers.conv2d(hm,
                              FLAGS.num_classes,
                              1,
                              stride=1,
                              padding='VALID',
                              activation_fn=None,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope='fc')
        hm = tf.sigmoid(hm)

    with tf.variable_scope('wh_layer', reuse=tf.AUTO_REUSE):
        wh = tc.layers.conv2d(logits,
                              256,
                              3,
                              stride=1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=None,
                              normalizer_params=None)

        wh = tc.layers.conv2d(wh,
                              2,
                              1,
                              stride=1,
                              padding='VALID',
                              activation_fn=None,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope='fc')

    with tf.variable_scope('off_layer', reuse=tf.AUTO_REUSE):
        off = tc.layers.conv2d(logits,
                              256,
                              3,
                              stride=1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=None,
                              normalizer_params=None)

        off = tc.layers.conv2d(off,
                              2,
                              1,
                              stride=1,
                              padding='VALID',
                              activation_fn=None,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope='fc')

    return hm, wh, off


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = tf.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = tf.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return tf.reduce_min((r1, r2, r3))


def draw_msra_gaussian(heatmap, center, sigma):
    origin_map=tf.identity(heatmap)
    tmp_size = sigma * 3
    mu_y = tf.cast(center[0] + 0.5, dtype=tf.float32)
    mu_x = tf.cast(center[1] + 0.5, dtype=tf.float32)
    h, w = tf.constant(int(heatmap.shape[0])), tf.constant(int(heatmap.shape[1]))
    ul = [tf.cast(mu_x - tmp_size, dtype=tf.int32), tf.cast(mu_y - tmp_size, dtype=tf.int32)]
    br = [tf.cast(mu_x + tmp_size + 1, dtype=tf.int32), tf.cast(mu_y + tmp_size + 1, dtype=tf.int32)]
    size = 2 * tmp_size + 1
    x = tf.range(0, size, 1, tf.float32)
    y = x[:, tf.newaxis]
    x0 = y0 = size // 2
    g = tf.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = tf.maximum(0, -ul[0]), tf.minimum(br[0], h) - ul[0]
    g_y = tf.maximum(0, -ul[1]), tf.minimum(br[1], w) - ul[1]
    img_x = tf.maximum(0, ul[0]), tf.minimum(br[0], h)
    img_y = tf.maximum(0, ul[1]), tf.minimum(br[1], w)
    img_x = tf.cast(img_x,dtype=tf.int32)
    img_y = tf.cast(img_y, dtype=tf.int32)

    img_mask_f=tf.ones_like(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]])

    paddings=[[img_y[0],h-img_y[1]],[img_x[0],w-img_x[1]]]

    img_mask=tf.pad(img_mask_f,paddings)

    origin_heatmap=heatmap*(1-img_mask)
    heat_max=tf.pad(tf.maximum(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]]),paddings)
    new_heatmap=origin_heatmap+heat_max

    check_hw = tf.logical_or(tf.greater_equal(ul[0], h),
                  tf.greater_equal(ul[1], w))
    check_br = tf.logical_or(tf.less(br[0], 0),
                  tf.less(br[1], 0))
    final_heatmap = tf.cond(tf.logical_or(check_hw,check_br),
                          lambda:origin_map,
                          lambda:new_heatmap)

    return final_heatmap


def generate_gt(boxes,labels,num_objects):
    num_class=FLAGS.num_classes
    input_size = FLAGS.image_size
    output_size = input_size//output_stride

    boxes_batch=tf.unstack(boxes, axis=0)
    labels_batch=tf.unstack(labels, axis=0)
    num_objcets_batch = tf.unstack(num_objects,axis=0)

    new_masks_batch=[]
    hw_mask_batch=[]
    offset_mask_batch=[]
    for boxes, labels, num_objects in zip(boxes_batch, labels_batch, num_objcets_batch):
        hm = tf.zeros([num_class, output_size, output_size])
        hw = tf.zeros([output_size, output_size, 2])
        offset = tf.zeros([output_size, output_size, 2])
        labels=labels-1
        def flows_body(new_mask, hw_mask, off_mask, boxes, labels, num_obj, checked_obj):
            cls_idx = tf.cast(labels[checked_obj],dtype=tf.int32)
            one_box = boxes[checked_obj, :] * output_size
            origin_one_box = boxes[checked_obj, :] * input_size
            origin_mask = new_mask[cls_idx,:,:]
            box_h = (one_box[2] - one_box[0])
            box_w = (one_box[3] - one_box[1])
            box_hw = (box_h, box_w)
            origin_box_c = ((origin_one_box[2] + origin_one_box[0]) / 2.,(origin_one_box[3] + origin_one_box[1]) / 2.)
            box_c = origin_box_c[0]//output_stride, origin_box_c[1]//output_stride
            box_c = tf.cast(box_c, dtype=tf.float32)
            box_off = origin_box_c[0]/output_stride - box_c[0], origin_box_c[1]/output_stride - box_c[1]

            new_mask_icls = tf.identity(origin_mask)
            sigma = gaussian_radius((box_h, box_w))

            hm_mask = draw_msra_gaussian(new_mask_icls, box_c, sigma)
            hm_non_zero = tf.cast(tf.greater(hm_mask, 0.), dtype=tf.float32)
            hm_center_mask = tf.cast(tf.equal(hm_mask, 1.), dtype=tf.float32)

            off_mask = off_mask *(1-tf.expand_dims(hm_center_mask,axis=-1))
            off_mask = off_mask + tf.ones([output_size, output_size, 2])*box_off*(tf.expand_dims(hm_center_mask,axis=-1))

            hw_mask = hw_mask *(1-tf.expand_dims(hm_center_mask,axis=-1))
            hw_mask = hw_mask + tf.ones([output_size, output_size, 2])*box_hw*(tf.expand_dims(hm_center_mask,axis=-1))

            new_mask_icls = new_mask_icls * tf.cast(1 - hm_non_zero, dtype=tf.float32) + hm_mask * tf.cast((hm_non_zero),dtype=tf.float32)
            new_mask_icls = tf.expand_dims(new_mask_icls,axis=0)
            front_stack=tf.zeros([cls_idx,output_size,output_size])
            end_stack = tf.zeros([num_class-(cls_idx+1), output_size, output_size])
            new_one_mask = tf.concat([front_stack, new_mask_icls, end_stack],axis=0)
            new_one_mask.set_shape([num_class,output_size,output_size])
            new_one_nonzero = tf.cast(tf.greater(new_one_mask,0),dtype=tf.float32)
            new_mask=new_mask*(1.-new_one_nonzero)
            final_mask=new_mask+new_one_mask
            return final_mask, hw_mask, off_mask, boxes, labels, num_obj, tf.add(checked_obj,1)


        def condition(new_mask, hw_mask, off_mask, boxes, labels, num_obj, checked_obj):
            return tf.less(checked_obj,num_obj)

        checked_objects = tf.constant(0, dtype = tf.int32)
        new_mask,hw_mask,off_mask,_,_,_,_=tf.while_loop(condition,flows_body,
                                    [hm,hw,offset,boxes,labels,num_objects,checked_objects],
                                    shape_invariants=[tf.TensorShape([num_class,output_size,output_size]),
                                                      tf.TensorShape([output_size, output_size, 2]),
                                                      tf.TensorShape([output_size, output_size, 2]),
                                                      boxes.get_shape(),
                                                      labels.get_shape(),
                                                      num_objects.get_shape(),
                                                      checked_objects.get_shape()])

        new_masks_batch.append(new_mask)
        hw_mask_batch.append(hw_mask)
        offset_mask_batch.append(off_mask)

    return tf.stack(new_masks_batch), tf.stack(hw_mask_batch), tf.stack(offset_mask_batch)

def _neg_loss(pred, wh, off, gt, wh_mask, off_mask):

    gt = tf.transpose(gt,[0,2,3,1])

    non_zero_wh=tf.cast(tf.greater(wh_mask,0.),dtype=tf.float32)
    non_zero_wh_pred = wh*non_zero_wh

    non_zero_off=tf.cast(tf.greater(off_mask,0.),dtype=tf.float32)
    non_zero_off_pred = off*non_zero_off

    wh_loss=tf.losses.huber_loss(wh_mask, non_zero_wh_pred,reduction="none")
    wh_loss=tf.reduce_sum(wh_loss)

    off_loss=tf.losses.huber_loss(off_mask, non_zero_off_pred,reduction="none")
    off_loss=tf.reduce_sum(off_loss)

    pos_inds = tf.cast(tf.equal(gt,1.),dtype=tf.float32)
    neg_inds = 1-pos_inds

    neg_weights = tf.pow(1. - gt, 4)

    loss = 0
    pred=tf.clip_by_value(pred,1E-10,1.0)
    pos_loss = tf.log(pred) * tf.pow(1. - pred, 2) * pos_inds
    neg_loss = tf.log(1. - pred) * tf.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = tf.reduce_sum(pos_inds)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    if num_pos == 0:
        hm_loss = loss - neg_loss
        wh_loss = wh_loss
        off_loss = off_loss
    else:
        hm_loss = loss - (pos_loss + neg_loss) / num_pos
        wh_loss = wh_loss / num_pos
        off_loss = off_loss / num_pos
    return hm_loss, wh_loss, off_loss, num_pos

def inference(images):
    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        train_model = model.resnet_18(is_training=False, input_size=FLAGS.image_size)

        logits = train_model._build_model(images)

        hm, wh, off = classifier(logits)

        return hm, wh, off

def loss(images, labels, boxes, num_objects):

    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

        train_model = model.resnet_18(is_training=True, input_size=FLAGS.image_size)

        logits = train_model._build_model(images)

        hm, wh, off = classifier(logits)

        label_mask,wh_mask,off_mask=generate_gt(boxes, labels, num_objects)

        hm_loss, wh_loss, off_loss, num_pos = _neg_loss(hm, wh, off, label_mask, wh_mask, off_mask)

    return hm_loss, wh_loss, off_loss, wh_mask, num_pos
