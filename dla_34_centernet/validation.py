# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""

import tensorflow as tf
import flags
FLAGS = tf.app.flags.FLAGS


import tensorflow as tf
import numpy as np

output_stride=4.

def non_max_sup(hm, kernel_size=3):
    hm_max = tf.nn.max_pool(hm, ksize=(1,kernel_size,kernel_size,1),strides=(1,1,1,1),padding='SAME')
    local_max = tf.cast(tf.equal(hm_max,hm),dtype=tf.float32)
    return hm * local_max


def top_k_(feat, k=20):
    gt_shape=tf.shape(feat)

    grid_h = tf.reshape(tf.range(gt_shape[1]),[-1,1,1])
    grid_w = tf.reshape(tf.range(gt_shape[2]),[1,-1,1])
    grid_cls = tf.reshape(tf.range(gt_shape[3]),[1,1,-1])

    grid_h = tf.tile(grid_h,[1,gt_shape[2],gt_shape[3]])
    grid_w = tf.tile(grid_w,[gt_shape[1],1,gt_shape[3]])
    grid_cls = tf.tile(grid_cls,[gt_shape[1],gt_shape[2],1])

    coordi=tf.stack([grid_h,grid_w,grid_cls],axis=-1)

    feat=tf.reshape(feat,[-1])
    _coordi=tf.reshape(coordi,[-1,3])
    top_k_flat=tf.nn.top_k(feat,k)
    coordi_gat=tf.gather(_coordi,top_k_flat[1],axis=0)
    coordi = coordi_gat[:,:2]
    classes = coordi_gat[:,2]
    scores = top_k_flat[0]
    return coordi, classes, scores

def decode_(heatmap, wh, offset):
    heatmap_ = non_max_sup(heatmap)
    coordi, classes, scores=top_k_(heatmap_,100)
    wh_ = tf.gather_nd(tf.squeeze(wh),coordi)
    offsets = tf.gather_nd(tf.squeeze(offset),coordi)
    box_center=tf.cast(coordi,dtype=tf.float32)+offsets
    boxes_y1x1=box_center-(wh_/2.)
    boxes_y2x2=box_center+(wh_/2.)
    boxes=tf.concat([boxes_y1x1,boxes_y2x2],axis=-1)/(tf.cast(FLAGS.image_size,dtype=tf.float32)/output_stride)
    return classes, scores, boxes, heatmap_


def one_image_validation(GT_loc,
                         GT_cls,
                         loc_pred,
                         idx_pred,
                         score_pred,
                         num_objects):

    GT_loc=np.squeeze(GT_loc)
    GT_cls = np.squeeze(GT_cls)
    loc_pred = np.squeeze(loc_pred)
    idx_pred = np.squeeze(idx_pred)
    score_pred = np.squeeze(score_pred)
    num_objects = np.squeeze(num_objects)

    def iou(boxes1,boxes2):
        if len(boxes1.shape)==1:
            boxes1=np.reshape(boxes1,[1,-1])

        if len(boxes2.shape)==1:
            boxes2=np.reshape(boxes2,[1,-1])

        tile_boxes1 = np.tile(np.expand_dims(boxes1, axis=0), [boxes2.shape[0], 1, 1])
        tile_boxes2 = np.tile(np.expand_dims(boxes2, axis=1), [1, boxes1.shape[0], 1])

        inter_y1 = np.maximum(tile_boxes1[:, :, 0], tile_boxes2[:, :, 0])
        inter_x1 = np.maximum(tile_boxes1[:, :, 1], tile_boxes2[:, :, 1])
        inter_y2 = np.minimum(tile_boxes1[:, :, 2], tile_boxes2[:, :, 2])
        inter_x2 = np.minimum(tile_boxes1[:, :, 3], tile_boxes2[:, :, 3])

        boxes1_area = (tile_boxes1[:, :, 2] - tile_boxes1[:, :, 0]) * (tile_boxes1[:, :, 3] - tile_boxes1[:, :, 1])
        boxes2_area = (tile_boxes2[:, :, 2] - tile_boxes2[:, :, 0]) * (tile_boxes2[:, :, 3] - tile_boxes2[:, :, 1])

        inter_width = inter_x2 - inter_x1
        inter_height = inter_y2 - inter_y1
        width_zero_mask = np.float32(inter_width > 0.)
        height_zero_mask = np.float32(inter_height > 0.)
        inter_width = inter_width * width_zero_mask
        inter_height = inter_height * height_zero_mask
        intersection_area = inter_width * inter_height

        iou = intersection_area / (boxes1_area + boxes2_area - intersection_area)

        return iou

    max_output=100
    total_max_output = 100
    nms_threshold=FLAGS.val_nms_threshold
    nms_score_threshold=1e-8


    loc_pred= np.array(loc_pred)

    clsPred_by_class=[]
    locPred_by_class=[]

    for k in range(FLAGS.num_classes):
        k_cls_idx = np.where(idx_pred==k)
        k_cls_score = score_pred[k_cls_idx]
        k_cls_loc = loc_pred[k_cls_idx, :]
        clsPred_by_class.append(k_cls_score)
        locPred_by_class.append(k_cls_loc)

    k_class_nms_loc = []
    k_class_nms_score = []

    for k_class_clsPred, k_class_locPred in zip(clsPred_by_class,locPred_by_class):

        score = np.reshape(k_class_clsPred, [-1])
        k_class_locPred = np.reshape(k_class_locPred, [-1, 4])
        score = score[np.where(score > nms_score_threshold)]
        sorted_idx = np.argsort(score)[::-1]
        sorted_loc = k_class_locPred[sorted_idx,:]
        sorted_score = score[sorted_idx]


        if len(sorted_loc) > max_output:
            sorted_loc = sorted_loc[:max_output]
            sorted_score = sorted_score[:max_output]

        valid_mask = np.ones(sorted_score.shape, dtype=np.int32)

        selected = []
        for k in range(len(valid_mask)):
            if k == 0:
                selected.append(k)
                selected = np.array(selected)
            else:
                iou_ = iou(sorted_loc[selected, :], sorted_loc[k, :])
                if np.max(iou_) < nms_threshold:
                    selected = np.concatenate([selected, np.array([k])],axis=0)

        valid_idx = selected
        k_class_nms_loc.append(sorted_loc[valid_idx])
        k_class_nms_score.append(sorted_score[valid_idx])


    # clip window and remove zero area

    for k_cls,(one_nms_score, one_nms_loc) in enumerate(zip(k_class_nms_score,k_class_nms_loc)):
        one_nms_loc[:, 0] = np.maximum(np.minimum(one_nms_loc[:, 0], 1.), 0.)
        one_nms_loc[:, 1] = np.maximum(np.minimum(one_nms_loc[:, 1], 1.), 0.)
        one_nms_loc[:, 2] = np.maximum(np.minimum(one_nms_loc[:, 2], 1.), 0.)
        one_nms_loc[:, 3] = np.maximum(np.minimum(one_nms_loc[:, 3], 1.), 0.)
        one_cls_area = (one_nms_loc[:,2] - one_nms_loc[:,0]) * (one_nms_loc[:,3] - one_nms_loc[:,1])
        non_zero_area_idx = np.where(one_cls_area > 0.)
        k_class_nms_score[k_cls] = one_nms_score[non_zero_area_idx]
        k_class_nms_loc[k_cls] = one_nms_loc[non_zero_area_idx]

    k_class_nms_idx = []
    for z, k_score in enumerate(k_class_nms_score):
        np_one=np.ones_like(k_score)
        class_idx=np_one*z
        k_class_nms_idx.append(class_idx)

    box_ = np.concatenate([box_ for box_ in k_class_nms_loc], 0)
    score_ = np.concatenate([score_ for score_ in k_class_nms_score], 0)
    nms_cls_idx= np.concatenate([cls_idx_ for cls_idx_ in k_class_nms_idx], 0)

    sort_all_nms_idx = np.argsort(score_)[::-1]
    cur_max_output=np.minimum(total_max_output,score_.shape[0])
    sorted_all_score = score_[sort_all_nms_idx][:cur_max_output]
    sorted_all_box = box_[sort_all_nms_idx][:cur_max_output]
    sorted_cls_idx = nms_cls_idx[sort_all_nms_idx][:cur_max_output]

    k_class_nms_score = []
    k_class_nms_loc = []

    for k in range(FLAGS.num_classes):
        k_cls_nms_idx = np.where(sorted_cls_idx==k)
        k_cls_nms_score = sorted_all_score[k_cls_nms_idx]

        k_cls_nms_loc = sorted_all_box[k_cls_nms_idx]
        k_class_nms_score.append(k_cls_nms_score)
        k_class_nms_loc.append(k_cls_nms_loc)

    #non maximum suppresion

    TF_array_by_class=[]
    TF_score_by_class=[]
    num_GT_by_class=[]
    GT_loc = GT_loc[:num_objects,:]
    GT_cls = GT_cls[:num_objects] - 1

    for k in range(FLAGS.num_classes):
        k_cls_index = np.where(GT_cls == k)
        num_GT_by_class.append(np.sum(np.int32(GT_cls == k)))

        if np.sum(np.int32(GT_cls == k))==0:
            TF_array_by_class.append(np.array([]))
            TF_score_by_class.append(np.array([]))
            continue
        k_cls_GT_loc = GT_loc[k_cls_index]

        TF_array = np.zeros([len(k_class_nms_score[k])])
        found_gt = np.zeros([np.sum(np.int32(GT_cls == k))])

        GT_pred_iou = iou(k_cls_GT_loc,k_class_nms_loc[k])

        maxiou_box_id=np.argmax(GT_pred_iou,axis=1)

        for x, TF in enumerate(TF_array):
            if np.max(GT_pred_iou[x,:]) > FLAGS.val_matching_threshold:
                if found_gt[maxiou_box_id[x]]==0:
                    TF_array[x]=1
                    found_gt[maxiou_box_id[x]]=1
        TF_array_by_class.append(TF_array)
        TF_score_by_class.append(k_class_nms_score[k])

    return TF_array_by_class, TF_score_by_class, num_GT_by_class

def compute_AP(entire_score,entire_TF,entire_numGT):

    sorted_entire_TF=[]
    for x_cls in range(FLAGS.num_classes):
        sorted_score_idx=np.argsort(entire_score[x_cls])[::-1]
        x_cls_TF=np.copy(entire_TF[x_cls])
        sorted_entire_TF.append(x_cls_TF[sorted_score_idx])

    entire_presicion = []
    entire_recall = []
    for num_GT, k_cls_TF in zip(entire_numGT, sorted_entire_TF):
        ac_sum = 0
        k_cls_precision = []
        k_cls_recall = []

        for num_pred, TF in enumerate(k_cls_TF):
            ac_sum += TF
            precision = ac_sum / (num_pred + 1)
            recall = ac_sum / (num_GT)
            k_cls_precision.append(precision)
            k_cls_recall.append(recall)

        for k_ in range(len(k_cls_precision))[::-1]:
            if k_== len(k_cls_precision)-1:
                continue
            k_cls_precision[k_]=np.maximum(k_cls_precision[k_],k_cls_precision[k_+1])
        entire_presicion.append(k_cls_precision)
        entire_recall.append(k_cls_recall)

    entire_AP = []
    entire_AP_sum = []

    for precision_k, recall_k in zip(entire_presicion, entire_recall):
        k_cls_AP_list=[]
        for i_, (precision_, recall_) in enumerate(zip(precision_k,recall_k)):
            if i_ == 0:
                AP_ = precision_*recall_
            else:
                AP_ = precision_*(recall_-recall_k[i_-1])
            k_cls_AP_list.append(AP_)

        k_cls_AP=np.sum(np.array(k_cls_AP_list))
        entire_AP.append(k_cls_AP_list)
        entire_AP_sum.append(k_cls_AP)


    return entire_AP_sum

