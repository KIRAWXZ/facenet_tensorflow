

import facenet
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import importlib
import numpy as np
from tensorflow.python import debug as tf_debug





def read_data():
    image_path_my = '/data/user_set/person_000138'
    dataset, _ = facenet.get_dataset(image_path_my)

    image_size = 224


    print("dataset", dataset)

    filenames = []
    for one in dataset:
        filenames = filenames + one.image_paths

    images = np.zeros((len(filenames), image_size, image_size, 3))

    print("filenames", filenames)
    for idx, filename in enumerate(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents)

        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

        # pylint: disable=no-member
        image.set_shape((224, 224, 3))
        images[idx, :, :, :] = image

    return images


def get_feature():
    network = importlib.import_module("models.nn2", 'inference')
    with tf.Graph().as_default():
        ops.reset_default_graph()
        images_placeholder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        batch_norm_params = {
            # Decay for the moving averages
            'decay': 0.995,
            # epsilon to prevent 0s in variance
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            # Only update statistics during training mode
            'is_training': phase_train_placeholder
        }
        # Build the inference graph
        prelogits, _ = network.inference(images_placeholder, 1, phase_train=phase_train_placeholder, weight_decay=1.0)
        pre_embeddings = slim.fully_connected(prelogits, 128, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              weights_regularizer=slim.l2_regularizer(0.0),
                                              normalizer_fn=slim.batch_norm,
                                              normalizer_params=batch_norm_params,
                                              scope='Bottleneck', reuse=False)
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3, allow_empty=True)
        # saver = tf.train.Saver(allow_empty=True)

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: False})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: False})

            saver.restore(sess,
                          "/home/chen/demo_dir/facenet_tensorflow_train/trained_model_2017_05_15_10_24/20170515-121856/model-20170515-121856.ckpt-182784")

            images = read_data()

            # Run forward pass to calculate embeddings
            print("images type", type(images), images)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            print(emb_array)


get_feature()