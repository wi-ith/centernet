# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange


import centernet
import input
import validation
import flags
import cv2


FLAGS = tf.app.flags.FLAGS

def draw_rectangle(image, p_class, p_score, p_box, height, width):
    put_text = '%d : %.1f' % (p_class, p_score * 100) + str('%')
    prediction_box = p_box * np.array(
        [height, width, height, width])
    prediction_box = prediction_box.astype(np.int32)
    p_class = int(p_class)
    if p_class % 1 == 0:
        cv2.rectangle(image, (prediction_box[1], prediction_box[0]),
                      (prediction_box[3], prediction_box[2]), (p_class*2, 200, p_class*2), 5)
        if prediction_box[2] < height*0.8:
            cv2.putText(image, put_text,
                        (prediction_box[1], prediction_box[2] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, put_text,
                        (prediction_box[1], prediction_box[2] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, name in grad_and_vars:
            if g==None:
               print(name)
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
            dtype=tf.float32)

        lr=FLAGS.learning_rate
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        with tf.name_scope('train_images'):
            images, labels, boxes, num_objects = input.distorted_inputs(FLAGS.batch_size)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
              [images, labels, boxes, num_objects], capacity=2 * FLAGS.num_gpus)

        tower_grads = []
        tower_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                        image_batch, label_batch, box_batch, num_objects_batch = batch_queue.dequeue()

                        loss_hm, loss_wh, loss_off, tmp_wh, tmp_pos = centernet.loss(image_batch, label_batch, box_batch, num_objects_batch)

                        loss = loss_hm + loss_wh*0.1 + loss_off

                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                        loss = loss + regularization_loss

                        tf.get_variable_scope().reuse_variables()

                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)
                        tower_losses.append(loss)

        grads = average_gradients(tower_grads)


        #validation
        val_images, val_labels, val_boxes, val_num_objects = input.inputs(1)
        with tf.device('/gpu:0'):
            with tf.name_scope('eval_images'):
              hm_pred, wh_pred, offset_pred = centernet.inference(val_images)
              classes_pred, scores_pred, boxes_pred, heatmap_pred = validation.decode_(hm_pred, wh_pred, offset_pred)


        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_images'))
        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, 'eval_images'))


        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))
        #
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        #
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        total_parameters=0
        trainable_list=[]
        for var in tf.trainable_variables():
            print(var.name)
            trainable_list.append(var.name)
            summaries.append(tf.summary.histogram(var.op.name, var))
            # shape is an array of tf.Dimension
            shape = var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('total_parameters : ',total_parameters)
        saver = tf.train.Saver(max_to_keep=20)

        summary_op = tf.summary.merge(summaries)

        pretrained_ckpt_path = FLAGS.pretrained_dir

        if tf.train.latest_checkpoint(FLAGS.ckpt_dir):
            print('use latest trained check point')
            init_fn = None
        else:
            if FLAGS.pretrained_dir=="":
                print('use no ckpt')
                init_fn = None
            else:
                print('use pretrained check point')
                variables_to_restore = slim.get_variables_to_restore(include=trainable_list,exclude=['global_step'])
                #        for k in variables_to_restore:
                #            print(k.name)
                init_fn = slim.assign_from_checkpoint_fn(FLAGS.pretrained_dir,
                                                         variables_to_restore, ignore_missing_vars=True)


        sv = tf.train.Supervisor(logdir=FLAGS.ckpt_dir,
                                 summary_op=None,
                                 saver=saver,
                                 save_model_secs=0,
                                 init_fn=init_fn)
        config_ = tf.ConfigProto(allow_soft_placement=True)
        config_.gpu_options.allow_growth = True
        # config_.gpu_options.per_process_gpu_memory_fraction = 1.0

        # sess=sv.managed_session(config=config_)
        with sv.managed_session(config=config_) as sess:
            # Start the queue runners.
            sv.start_queue_runners(sess=sess)

            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                sess.run(train_op)

                sv_global_step, loss_value, loss_hm_, loss_wh_, loss_off_ = sess.run([sv.global_step,loss, loss_hm, loss_wh, loss_off])

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if sv_global_step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus
                    epochs = sv_global_step * FLAGS.batch_size / FLAGS.num_train

                    format_str = ('epochs %.2f, step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                    print (format_str % (epochs, step, loss_value,
                                         examples_per_sec, sec_per_batch))
                    print('loss_hm : ', loss_hm_, 'loss_wh : ',loss_wh_,'loss_off : ', loss_off_)

                if sv_global_step % 10 == 0:
                    summary_str = sess.run(summary_op)
                    sv.summary_computed(sess, summary_str)

                if sv_global_step % (int(FLAGS.num_train / FLAGS.batch_size)*1) == 0 and sv_global_step!=0:

                    print('start validation')
                    entire_TF=[]
                    entire_score=[]
                    entire_numGT=[]
                    for val_step in range(FLAGS.num_validation):

                        if val_step%500==0:
                            print(val_step,' / ',FLAGS.num_validation)

                        val_img, \
                        val_GT_boxes, \
                        val_GT_cls, \
                        val_loc_pred, \
                        val_cls_pred, \
                        val_score_pred, \
                        num_objects = sess.run([val_images,
                                                val_boxes,
                                                val_labels,
                                                boxes_pred,
                                                classes_pred,
                                                scores_pred,
                                                val_num_objects])

                        TF_array, TF_score, num_GT = validation.one_image_validation(val_GT_boxes,
                                                                                     val_GT_cls,
                                                                                     val_loc_pred,
                                                                                     val_cls_pred,
                                                                                     val_score_pred,
                                                                                     num_objects)

                        if len(entire_TF) == 0:
                            entire_TF = TF_array
                            entire_score = TF_score
                            entire_numGT = num_GT
                        else:
                            for k_cls in range(FLAGS.num_classes-1):
                                entire_TF[k_cls]=np.concatenate([entire_TF[k_cls],TF_array[k_cls]],axis=0)
                                entire_score[k_cls]=np.concatenate([entire_score[k_cls],TF_score[k_cls]],axis=0)
                                entire_numGT[k_cls]+=num_GT[k_cls]

                    entire_AP_sum = validation.compute_AP(entire_score,entire_TF,entire_numGT)

                    mAP = np.sum(np.array(entire_AP_sum)) / np.sum(np.array(entire_AP_sum) != 0)

                    print('class AP : ',entire_AP_sum)
                    print(len(entire_AP_sum))
                    print('mAP : ',mAP)

                    checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=sv.global_step)




def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
