# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from lib.core.model.sqeeze_excitation.se import se
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish

class CenternetHead():

    def __call__(self, fms, training=True):
        arg_scope = resnet_arg_scope( bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5


                deconv_feature = self._unet_magic(fms)

                #####
                kps = slim.separable_conv2d(deconv_feature,
                                  cfg.DATA.num_class,
                                  [3, 3],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(-2.19),
                                  scope='centernet_cls_output')

                wh = slim.separable_conv2d(deconv_feature,
                                 4,
                                 [3, 3],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0),
                                 scope='centernet_wh_output')

        return kps, wh*16



    def _unet_magic(self, fms, dim=96):
        c2, c3, c4, c5 = fms

        ###fm 32
        c5 = slim.conv2d(c5,
                         dim,
                         [1, 1],
                         activation_fn=None,
                         padding='SAME',
                         scope='c5_1x1')
        c5_upsample=tf.keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c5)
        c4 = slim.conv2d(c4,
                         dim,
                         [1, 1],
                         activation_fn=None,
                         padding='SAME', scope='c4_1x1')

        p4 = hard_swish(c4 + c5_upsample)

        p4_upsample = tf.keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(p4)
        p4_upsample = slim.conv2d(p4_upsample,
                         dim,
                         [1, 1],
                         activation_fn=None,
                         padding='SAME', scope='p4_1x1')

        c3 = slim.conv2d(c3,
                         dim,
                         [1, 1],
                         activation_fn=None,
                         padding='SAME', scope='c3_1x1')

        p3 = hard_swish(c3 + p4_upsample)

        p3_upsample = tf.keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(p3)
        p3_upsample = slim.conv2d(p3_upsample,
                                  dim,
                                  [1, 1],
                                  activation_fn=None,
                                  padding='SAME', scope='p3_1x1')
        c2 = slim.conv2d(c2,
                         dim,
                         [1, 1],
                         activation_fn=None,
                         padding='SAME', scope='c2_1x1')
        combine_fm = hard_swish(c2 + p3_upsample)

        return combine_fm