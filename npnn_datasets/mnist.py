
import os
import numpy as np
from npnn_datasets import DataSet
from PIL import Image

class MNIST(DataSet):

    def __init__(self, points=None, train_fraction=0.7):
        super(MNIST, self).__init__()

        data_dir = os.path.join(os.path.dirname(__file__), 'mnist')

        train, test = MNIST_DataLoader().read_data_sets(data_dir=data_dir, one_hot=True)

        self.x_data = train.images
        self.y_data = train.labels

        #self.save_png_files()

        self.prepare(train_fraction, normalize_x=True, normalize_y=False)


    def save_png_files(self):
        for n in np.arange(self.x_data.shape[0]):
            img = Image.fromarray(np.array(self.x_data[n]).reshape((28, 28)))
            for c in np.arange(10):
                if self.y_data[n][c] > 0.5:
                    img.save("mnist_28x28/mnist_{0}_{1:05d}.png".format(c, n))
                    img = img.resize((14, 14), resample=Image.BILINEAR)
                    img.save("mnist_14x14/mnist_{0}_{1:05d}.png".format(c, n))
        exit()


# ==============================================================================
# the following python code is an adapted version of:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
# ==============================================================================

# ==============================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gzip

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

class MNIST_DataLoader:

  def _read32(self, bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


  def _extract_images(self, f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    with gzip.GzipFile(fileobj=f) as bytestream:
      magic = self._read32(bytestream)
      if magic != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
      num_images = self._read32(bytestream)
      rows = self._read32(bytestream)
      cols = self._read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      data = data.reshape(num_images, rows, cols, 1)
      return data


  def _dense_to_one_hot(self, labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


  def _extract_labels(self, f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
      magic = self._read32(bytestream)
      if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
      num_items = self._read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      if one_hot:
        return self._dense_to_one_hot(labels, num_classes)
      return labels


  def read_data_sets(self, data_dir='', one_hot=False, dtype=dtypes.float32, reshape=True):

    train_images_file = 'train-images-idx3-ubyte.gz'
    train_labels_file = 'train-labels-idx1-ubyte.gz'
    test_images_file = 't10k-images-idx3-ubyte.gz'
    test_labels_file = 't10k-labels-idx1-ubyte.gz'

    train_images_file = os.path.join(data_dir, train_images_file)
    train_labels_file = os.path.join(data_dir, train_labels_file)
    test_images_file  = os.path.join(data_dir, test_images_file)
    test_labels_file  = os.path.join(data_dir, test_labels_file)

    with gfile.Open(train_images_file, 'rb') as f:
      train_images = self._extract_images(f)

    with gfile.Open(train_labels_file, 'rb') as f:
      train_labels = self._extract_labels(f, one_hot=one_hot)

    with gfile.Open(test_images_file, 'rb') as f:
      test_images = self._extract_images(f)

    with gfile.Open(test_labels_file, 'rb') as f:
      test_labels = self._extract_labels(f, one_hot=one_hot)

    options = dict(dtype=dtype, reshape=reshape)

    train = MNIST_DataSet(train_images, train_labels, **options)
    test = MNIST_DataSet(test_images, test_labels, **options)

    return train, test

class MNIST_DataSet:

  def __init__(self, images, labels, dtype=dtypes.float32, reshape=True):
    """Construct a MNIST_DataSet.

    `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.

    Args:
      images: The images
      labels: The labels
      dtype: Output image dtype. One of [uint8, float32]. `uint8` output has
        range [0,255]. float32 output has range [0,1].
      reshape: Bool. If True returned images are returned flattened to vectors.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype

    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)
      )

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

