import tensorflow as tf


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def read_examples_list(path):
  """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip() for line in lines]



def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not len(xml):
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')


def load_labelmap(path):
  """Loads label map.

  Args:
    path: path to label map text file.
  Returns:
    a list that each class name and id
  """
  label_map = open(path, 'r')
  all_labels = list(label_map.read().splitlines())
  labels = []
  for label in all_labels:
    split = label.split(' ')
    id = int(split[0])
    txt = split[1]
    merge = (id, txt)
    labels.append(merge)
  return labels


def get_label_map_dict(label_map_path):
  """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path: path to label_map.

  Returns:
    A dictionary mapping label names to id.
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map:
    label_map_dict[item[1]] = item[0]
  return label_map_dict


def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords
