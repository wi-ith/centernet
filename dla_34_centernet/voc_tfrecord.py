from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import os
import io
import random

from lxml import etree
import PIL.Image
import tensorflow as tf
import tfrecord_utils


logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path To Image Directory.')
flags.DEFINE_string('annotations_dir', '', 'Path To XML Directory.')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('file_list_txt', '', 'Image File List')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data, channels=3):
        if not channels == 3:
            raise ValueError('channels must be 3')
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image


def dict_to_tf_example(data,
                       image_file_name,
                       image_directory,
                       label_map_dict,
                       coder):
    img_name = image_file_name + '.jpg'
    full_path = os.path.join(image_directory, img_name)
    if not tf.gfile.Exists(full_path):
        full_path = os.path.join(full_path[:-3]+'jpeg')
    if tf.gfile.Exists(full_path) != 1:
        print('1')
        return 0
    encoded_jpg = tf.gfile.GFile(full_path, 'rb').read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    if width == 0 or height == 0:
        print('2')
        return 0

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    if 'object' in data:
        for obj in data['object']:
            obj['name'] = obj['name'].lower()

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)

            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

            if len(classes) == 0:
                return 0
            elif len(classes) != len(classes_text):
                return 0
            elif len(classes) != len(xmin):
                return 0
              
    if len(classes) >= 100:
      print('This image has more than 100 objects :', image_file_name)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_utils.int64_feature(height),
        'image/width': tfrecord_utils.int64_feature(width),
        'image/filename': tfrecord_utils.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': tfrecord_utils.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': tfrecord_utils.bytes_feature(key.encode('utf8')),
        'image/encoded': tfrecord_utils.bytes_feature(encoded_jpg),
        'image/format': tfrecord_utils.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': tfrecord_utils.float_list_feature(xmin),
        'image/object/bbox/xmax': tfrecord_utils.float_list_feature(xmax),
        'image/object/bbox/ymin': tfrecord_utils.float_list_feature(ymin),
        'image/object/bbox/ymax': tfrecord_utils.float_list_feature(ymax),
        'image/object/class/text': tfrecord_utils.bytes_list_feature(classes_text),
        'image/object/class/label': tfrecord_utils.int64_list_feature(classes)
        }))

    return example


def main(_):
    image_dir = FLAGS.image_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = tfrecord_utils.get_label_map_dict(FLAGS.label_map_path)
    logging.info('Start createing TFRecord.')
    annotations_dir = FLAGS.annotations_dir
    examples_list = tfrecord_utils.read_examples_list(FLAGS.file_list_txt)
    random.shuffle(examples_list)
    coder = ImageCoder()
    in_record_images = 0
    out_record_images = 0
    for i, example in enumerate(examples_list):
        if i % 100 == 0:
            logging.info('Done %d / %d / In : %d / Out : %d', i, len(examples_list), in_record_images,
                         out_record_images)
        path = os.path.join(annotations_dir, example + '.xml')
        if not tf.gfile.Exists(path):
            out_record_images += 1
            continue
        with tf.gfile.GFile(path, 'rb') as fid:
            xml_str = fid.read()
            xml = etree.fromstring(xml_str)
        try:
            data = tfrecord_utils.recursive_parse_xml_to_dict(xml)['Annotation']
        except KeyError:
            data = tfrecord_utils.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, example, image_dir, label_map_dict, coder)
        if tf_example == 0:
            out_record_images += 1
            continue
        in_record_images += 1
        writer.write(tf_example.SerializeToString())

    logging.info('number of image in TFRecord : %d', in_record_images)
    logging.info('number of image out TFRecord : %d', out_record_images)

if __name__ == '__main__':
  tf.app.run()
