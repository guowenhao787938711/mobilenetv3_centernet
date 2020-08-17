# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from lib.core.model.sqeeze_excitation.se import se

class CenternetHead():

    def __call__(self, fms, training=True):
        arg_scope = resnet_arg_scope( bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5


                deconv_feature = self._unet_magic(fms)

                #####
                kps = slim.conv2d(deconv_feature,
                                  cfg.DATA.num_class,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(-2.19),
                                  scope='centernet_cls_output')


                wh = slim.conv2d(deconv_feature,
                                 4,
                                 [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0),
                                 scope='centernet_wh_output')

        return kps, wh*16

    def _complex_upsample(self,fm,output_dim, factor=2,scope='upsample'):
        with tf.variable_scope(scope):


            x = slim.separable_conv2d(fm,
                                       output_dim//2,
                                       [3, 3],
                                       activation_fn=None,
                                       padding='SAME',
                                       scope='branch_x_upsample_resize')
            y = slim.separable_conv2d(fm,
                                       output_dim//2,
                                       [5, 5],
                                       activation_fn=None,
                                       padding='SAME',
                                       scope='branch_y_upsample_resize')
            final = x+y
            final = tf.keras.layers.UpSampling2D(data_format='channels_last', interpolation='bilinear',
                                                          size=(factor, factor))(final)

            return final

    def _unet_magic(self, fms, dims=cfg.MODEL.head_dims):

        c2, c3, c4, c5 = fms

        ####24, 116, 232, 256,

        c5_upsample = self._complex_upsample(c5, output_dim= dims[0],factor=8, scope='c5_upsample')

        c4_upsample = self._complex_upsample(c4, output_dim= dims[1], factor=4,scope='c4_upsample')

        c3_upsample = self._complex_upsample(c3, output_dim= dims[2],factor=2, scope='c3_upsample')

        final = tf.concat([c5_upsample,c4_upsample,c3_upsample,c2],axis=3)

        return final

