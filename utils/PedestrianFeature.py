#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: PedestrianFeature.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-04-04
"""

import _init_paths
import yaml
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import os

class PedestrianFeature(object):

    def __init__(self, configfile):

        # wide resnet pedestrian feature detect config init
        tfconfig = yaml.load(open(configfile))
        self.image_shape = tuple(tfconfig['image_shape'])
        self.tfmodel = tfconfig['tfmodel']
        self.batch_size = tfconfig['batch_size']
        self.loss_mode = tfconfig['loss_mode']
        self.pre_feature_output = tfconfig['pre_feature_output']
        self.GPU_fraction = tfconfig['GPU_fraction']
        self.pre_objdetect_output = tfconfig['pre_objdetect_output']

        # tensorflow session config set
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.GPU_fraction)
        #self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # tensorflow placeholder
        #self.image_var = tf.placeholder(tf.uint8, (None,) + image_shape)
        # tensorflow session and reset init
        self.encoder = self._feature_detect_func()

    def _batch_norm_fn(self, x):
        return slim.batch_norm(x, tf.get_variable_scope().name + "/bn")

    def _create_inner_block(self, incoming, scope):
       
        nonlinearity = tf.nn.elu
        weights_initializer = tf.truncated_normal_initializer(1e-3)
        bias_initializer = tf.zeros_initializer()
        regularizer = None
        increase_dim = False
        summarize_activations = True
        n = incoming.get_shape().as_list()[-1]
        stride = 1
        if increase_dim:
            n *= 2
            stride = 2

        incoming = slim.conv2d(
            incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
            normalizer_fn = self._batch_norm_fn, weights_initializer=weights_initializer,
            biases_initializer=bias_initializer, weights_regularizer=regularizer,
            scope=scope + "/1")
        if summarize_activations:
            tf.summary.histogram(incoming.name + "/activations", incoming)
        
        incoming = slim.dropout(incoming, keep_prob=0.6)
        
        incoming = slim.conv2d(
            incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
            normalizer_fn=None, weights_initializer=weights_initializer,
            biases_initializer=bias_initializer, weights_regularizer=regularizer,
            scope=scope + "/2")
        return incoming

    def _residual_block(self, incoming, scope, nonlinearity=tf.nn.elu,
            weights_initializer=tf.truncated_normal_initializer(1e-3),
            bias_initializer=tf.zeros_initializer(), regularizer=None,
            increase_dim=False, is_first=False,
            summarize_activations=True):

        if is_first:
            network = incoming
        else:
            network = self._batch_norm_fn(incoming)
            network = nonlinearity(network)
            if summarize_activations:
                tf.summary.histogram(scope + "/activations", network)

        pre_block_network = network
        post_block_network = self._create_inner_block(pre_block_network, scope)

        incoming_dim = pre_block_network.get_shape().as_list()[-1]
        outgoing_dim = post_block_network.get_shape().as_list()[-1]
        if incoming_dim != outgoing_dim:
            assert outgoing_dim == 2 * incoming_dim, \
                "%d != %d" % (outgoing_dim, 2 * incoming)
            projection = slim.conv2d(
                incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
                scope=scope + "/projection", weights_initializer=weights_initializer,
                biases_initializer=None, weights_regularizer=regularizer)
            network = projection + post_block_network
        else:
            network = incoming + post_block_network
        return network

    def _create_network(self, incoming, num_classes, reuse=None, l2_normalize=True,
            weight_decay=1e-8):

        nonlinearity = tf.nn.elu
        # conv layer
        conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
        conv_bias_init = tf.zeros_initializer()
        conv_regularizer = slim.l2_regularizer(weight_decay)
        # full layer
        fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
        fc_bias_init = tf.zeros_initializer()
        fc_regularizer = slim.l2_regularizer(weight_decay)

        network = incoming

        network = slim.conv2d(
            network, 32, [3, 3], stride=1, activation_fn=tf.nn.elu,
            padding="SAME", normalizer_fn = self._batch_norm_fn, scope="conv1_1",
            weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
            weights_regularizer=conv_regularizer)
    	if True:
    	    tf.summary.histogram(network.name + "/activations", network)
    	    tf.summary.image("conv1_1/weights", tf.transpose(
    	        slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]), 128)

        network = slim.conv2d(
            network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
            padding="SAME", normalizer_fn = self._batch_norm_fn, scope="conv1_2",
            weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
            weights_regularizer=conv_regularizer)
    	if True:
    	    tf.summary.histogram(network.name + "/activations", network)

        network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

        network = self._residual_block(
            network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False, is_first=True,
            summarize_activations=True)
        network = self._residual_block(
            network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=True)
        network = self._residual_block(
            network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=True,
            summarize_activations=True)
        network = self._residual_block(
            network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=True)
        network = self._residual_block(
            network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=True,
            summarize_activations=True)
        network = self._residual_block(
            network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=True)

        feature_dim = network.get_shape().as_list()[-1]
        print("feature dimensionality: ", feature_dim)
        network = slim.flatten(network)

        network = slim.dropout(network, keep_prob=0.6)
        network = slim.fully_connected(
            network, feature_dim, activation_fn=nonlinearity,
            normalizer_fn = self._batch_norm_fn, weights_regularizer=fc_regularizer,
            scope="fc1", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

        features = network

        if l2_normalize:
            # Features in rows, normalize axis 1.
            features = slim.batch_norm(features, scope="ball", reuse=reuse)
            feature_norm = tf.sqrt(
                tf.constant(1e-8, tf.float32) +
                tf.reduce_sum(tf.square(features), [1], keep_dims=True))
            features = features / feature_norm

            with slim.variable_scope.variable_scope("ball", reuse=reuse):
                weights = slim.model_variable(
                    "mean_vectors", (feature_dim, num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=1e-3),
                    regularizer=None)
                scale = slim.model_variable(
                    "scale", (num_classes,), tf.float32,
                    tf.constant_initializer(0., tf.float32), regularizer=None)
                if True:
                    tf.summary.histogram("scale", scale)
                scale = tf.nn.softplus(scale)

            # Each mean vector in columns, normalize axis 0.
            weight_norm = tf.sqrt(
                tf.constant(1e-8, tf.float32) +
                tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
            logits = scale * tf.matmul(features, weights / weight_norm)

        else:
            logits = slim.fully_connected(
                features, num_classes, activation_fn=None,
                normalizer_fn=None, weights_regularizer=fc_regularizer,
                scope="softmax", weights_initializer=fc_weight_init,
                biases_initializer=fc_bias_init)

        return features, logits


    def tffunc(x):
        self.session.run(feature_var, feed_dict=x)

    def _run_in_batches(func, data_dict, output):
        data_len = len(output)
        num_batches = int(data_len / self.batch_size)

        start, end = 0, 0
        for index in range(num_batches):
            start, end = index * self.batch_size, (index + 1) * self.batch_size
            batch_data_dict = {key: value[start:end] for key,value in data_dict.items()}
            output[start:end] = func(batch_data_dict)

        if end < len(output):
            batch_data_dict = {key: value[end:] for key,value in data_dict.items()}
            output[end:] = func(batch_data_dict)

    def _factory_fn(self, image, reuse, l2_normalize):
        num_classes = 1501
        is_training = False
        weight_decay = 1e-8
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                 slim.batch_norm, slim.layer_norm],
                                reuse=reuse):
                features, logits = self._create_network(
                    image, num_classes, None, True, weight_decay)
                return features, logits

    def _preprocess_fn(self, image):
        image = image[:, :, ::-1]  # BGR to RGB
        return image

    def _create_image_encoder(self, model_filename):

        image_var = tf.placeholder(tf.uint8, (None,) + self.image_shape)

        preprocessed_image_var = tf.map_fn(
            lambda x: self._preprocess_fn(x),
            tf.cast(image_var, tf.float32))

        l2_normalize = self.loss_mode == "cosine"
        feature_var, _ = self._factory_fn(preprocessed_image_var, None, l2_normalize)
        feature_dim = feature_var.get_shape().as_list()[-1]

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        def encoder(data_x):
            out = np.zeros((len(data_x), feature_dim), np.float32)
            _run_in_batches(
                lambda x: session.run(feature_var, feed_dict=x),
                {image_var: data_x}, out, self.batch_size)
            return out

        return encoder


    def _feature_detect_func(self):
        image_encoder = self._create_image_encoder(self.tfmodel)

        def encoder(image, boxes):
            image_patches = []
            for box in boxes:
                patch = extract_image_patch(image, box, self.image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(
                        0., 255., self.image_shape).astype(np.uint8)
                image_patches.append(patch)
            image_patches = np.asarray(image_patches)
            return image_encoder(image_patches)

        return encoder

    def generate_pedestrian_feature(self, input):
        '''
        param: input (imagedics list, input_path)
            use Pedestrian feature detect function get features
        '''
        image_dics = input[0]
        input_path = input[1]

        path_split = input_path.split('/')
        time_stamp = path_split[-1].split('.')[0]

        # output_dir = pre + shopname + data + camera + Append
        output_dir = self.pre_feature_output + path_split[1] + '/' + path_split[2] + '/' + 'Channel_01' + '/Append'
        
        # txt_dir = pre + shopname + data + camera + Append
        objdetect_input_dir = self.pre_objdetect_output + '/detOutput/' + path_split[1] + '/' + path_split[2] + '/' + 'Channel_01' + '/Append'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
       
        detection_file = os.path.join(objdetect_input_dir, time_stamp + '.txt') 
        output_filename = os.path.join(output_dir, time_stamp + '.npy')
        image_filenames = {}
        detections_out = []
        
        if not os.path.exists(detection_file):
            np.save(
                output_filename, np.asarray(detections_out), allow_pickle=False)
            return

        detections_in = np.loadtxt(detection_file, delimiter=',')
        if len(detections_in.shape) < 2:
            np.save(
                output_filename, np.asarray(detections_out), allow_pickle=False)
            return

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        
            mask = frame_indices == frame_idx
            rows = detections_in[mask]
            if frame_idx > 40 or frame_idx < 1:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = image_dics[frame_idx-1]
            features = self.encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                            in zip(rows, features)]

        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)
