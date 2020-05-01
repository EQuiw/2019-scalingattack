import numpy as np
import typing
import tensorflow as tf

from scaling.ScalingApproach import ScalingApproach
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms



class TFImageScaler(ScalingApproach):

    def __init__(self,
                 algorithm: typing.Union[int, SuppScalingAlgorithms],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 target_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):

        super().__init__(algorithm, src_image_shape, target_image_shape)


    def scale_image_with(self, xin: np.ndarray, trows: int, tcols: int):

        prev_shape = xin.shape
        prev_dtype = xin.dtype

        # We create here a new graph (instead of using the default graph), otherwise each scaling operation
        # will add some operations to the graph, making the graph slower and slower. In this way, the default
        # graph can be easily used for our neural network model.
        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            # create shape (batch=1, rows, cols, channels).
            if len(xin.shape) == 2:
                xin = np.expand_dims(xin, axis=2)

            x = tf.placeholder(tf.float32, shape=[None] + list(xin.shape))
            x_scaled = tf.image.resize_images(x, (trows, tcols), method=self.algorithm)

            xin = np.expand_dims(xin, axis=0)
            src_out = sess.run(x_scaled, feed_dict={x: xin})

            if len(prev_shape) == 2:
                src_ = src_out.reshape((trows, tcols))
            else:
                src_ = src_out.reshape((trows, tcols, prev_shape[2]))

            # we want to have the same output type as input type; astype rounds towards zero, so we use round before.
            if np.issubdtype(prev_dtype, np.integer):
                src_ = np.round(src_).astype(prev_dtype)

            return src_


    def _convert_suppscalingalgorithm(self, algorithm: SuppScalingAlgorithms):
        if algorithm == SuppScalingAlgorithms.NEAREST:
            return tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif algorithm == SuppScalingAlgorithms.LINEAR:
            return tf.image.ResizeMethod.BILINEAR
        elif algorithm == SuppScalingAlgorithms.CUBIC:
            return tf.image.ResizeMethod.BICUBIC
        elif algorithm == SuppScalingAlgorithms.LANCZOS:
            raise NotImplementedError()
        elif algorithm == SuppScalingAlgorithms.AREA:
            return tf.image.ResizeMethod.AREA
        else:
            raise NotImplementedError()