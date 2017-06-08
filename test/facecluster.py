#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import glob
import align.detect_face
import facenet
import bktree
import shutil
import json

from PIL import Image
from scipy import misc
from scipy import ndimage
from scipy import spatial

import importlib
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np

w_EXIFROTATE_ = {1:0, 3:180, 6:-90, 8:90}
w_THRESHOLD_SCORE_ = 0.99
w_MARGIN_OF_CORP_ = 44
w_MODEL_PATH_ = "../models/tensorflow/20170216-091149"
w_HEIGHT_WIDTH_ = 10
w_SIMILAR_ = 10
w_TORCH_IMAGE_SIZE_ = 96

def dhash(image, hash_size=8):
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    pixels = np.array(image.getdata(), dtype=np.float).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff

def calc_abs_rel_position(bounding_boxes, image_size):
    absposition = np.zeros([bounding_boxes.shape[0], 4], dtype=np.int)
    relposition = np.zeros([bounding_boxes.shape[0], 4], dtype=np.int)

    absposition[:, 0] = np.maximum(bounding_boxes[:, 0] - w_MARGIN_OF_CORP_ / 2, 0)
    absposition[:, 1] = np.maximum(bounding_boxes[:, 1] - w_MARGIN_OF_CORP_ / 2, 0)
    absposition[:, 2] = np.minimum(bounding_boxes[:, 2] + w_MARGIN_OF_CORP_ / 2, image_size[1])
    absposition[:, 3] = np.minimum(bounding_boxes[:, 3] + w_MARGIN_OF_CORP_ / 2, image_size[0])

    relposition[:, 0] = absposition[:, 0] / image_size[1] * 100
    relposition[:, 1] = absposition[:, 1] / image_size[0] * 100
    relposition[:, 2] = absposition[:, 2] / image_size[1] * 100
    relposition[:, 3] = absposition[:, 3] / image_size[0] * 100

    return absposition, relposition

def compareDHash(src, dst):
    return np.count_nonzero(src.dhash != dst.dhash)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def WPictureWriter(wp_list):
    ret_list = []
    for wp in wp_list:
        ret_list.append(wp.toDict())

    return json.dumps(ret_list)

def WPictureReader(wp_dict):
    wp = WPicture()
    wp.fpath = wp_dict['fpath']
    wp.key = wp_dict['key']
    wp.rotate = wp_dict['rotate']
    wp.image_size = np.array(wp_dict['image_size'], dtype=np.int)

    img = misc.imread(wp.fpath)
    img = misc.imresize(img, wp.image_size, interp='bilinear')
    img = ndimage.rotate(img, wp.rotate)
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]
    wp.img = img
    wp.bounding_boxes = np.array(wp_dict['bounding_boxes'], dtype = np.float)
    wp.absposition = np.array(wp_dict['absposition'], dtype = np.int)
    wp.relposition = np.array(wp_dict['relposition'], dtype = np.int)
    wp.dhash = np.array(wp_dict['dhash'], dtype = np.bool)

    return wp


class WPicture:
    def __init__(self, file_path=None):
        self.fpath = file_path
        self.key = self.fpath
        self.rotate = 0
        self.img = None
        self.image_size = None
        self.bounding_boxes = None
        self.absposition = None
        self.relposition = None
        self.dhash = None

        if file_path is not None:
            with Image.open(file_path) as im:
                try:
                    exif_data = im._getexif()
                    self.rotate = w_EXIFROTATE_[exif_data[274]] if exif_data else 0
                except:
                    self.rotate = 0
                self.dhash = dhash(im).reshape((64))

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __lt__(self, other):
        return self.key < other.key

    def __gt__(self, other):
        return self.key >= other.key

    def __hash__(self):
        return self.key.__hash__()

    def __str__(self):
        return self.key.__str__()

    def __repr__(self):
        return self.key.__repr__()

    def toDict(self):
        ret_dict = {}
        ret_dict['fpath'] = self.fpath
        ret_dict['key'] = self.key
        ret_dict['rotate'] = self.rotate
        ret_dict['image_size'] = self.image_size.tolist()
        ret_dict['bounding_boxes'] = self.bounding_boxes.tolist()
        ret_dict['absposition'] = self.absposition.tolist()
        ret_dict['relposition'] = self.relposition.tolist()
        ret_dict['dhash'] = self.dhash.tolist()
        return ret_dict

    def setBoundingBox(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes

    def getFaceCount(self):
        return self.bounding_boxes.shape[0] if self.bounding_boxes is not None else 0

    def getFaceScore(self):
        return self.bounding_boxes[:, -1] if self.bounding_boxes is not None else None

    def getFaceAbsPosition(self):
        return self.absposition

    def getFaceRelPosition(self):
        return self.relposition

    def setFaceAbsPosition(self, absposition):
        self.absposition = absposition

    def setFaceRelPosition(self, relposition):
        self.relposition = relposition


class WPictureGroup:
    def __init__(self, wpset):
        if type(wpset) is not list:
            raise TypeError("wpset must be a list")
        if np.count_nonzero([type(w) is not WPicture for w in wpset]) != 0:
            raise TypeError("the elements of wpset must be WPicture")
        self.wpicture_set = wpset
        self.delegate = None
        self.embeddings = None
        self.findTheBestFace()

    def __len__(self):
        return len(self.wpicture_set)

    def __getitem__(self, idx):
        return self.wpicture_set[idx]

    def __str__(self):
        return self.wpicture_set.__str__()

    def __repr__(self):
        return self.wpicture_set.__repr__()

    def append(self, wp):
        if type(wp) is not WPicture:
            raise TypeError("wp must be WPicture")
        self.wpicture_set.append(wp)
        self.findTheBestFace()

    def deleteByIdx(self, idx):
        del(self.wpicture_set[idx])

    def deleteByObj(self, obj):
        self.wpicture_set.remove(obj)

    def deleteByKey(self, key):
        for w in self.wpicture_set:
            if w.key == key:
                deleteByObj(w)

    def getGroupDelegate(self):
        if self.delegate is None:
            raise RuntimeError("This group have no object")
        return self.wpicture_set[self.delegate]

    def findTheBestFace(self):
        if len(self) == 0:
            return None
        count_list = [w.getFaceCount() for w in self.wpicture_set]
        maxval = max(count_list)
        maxval_idx = [i for i,k in enumerate(count_list) if k == maxval]
        scores = np.zeros([len(maxval_idx), maxval], dtype=np.float)
        for i,v in enumerate(maxval_idx):
            scores[i,:] = self.wpicture_set[v].getFaceScore()
        posi = np.sum(scores, axis=1).argmax()
        self.delegate = maxval_idx[posi]

    def setFeatureForDelegate(self, emb):
        self.embeddings = emb

def detectFace(files_list, args):

    ret_wp_list = []
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    noface_dir = os.path.join(args.input_dir, "noface")
    if os.path.exists(noface_dir):
        shutil.rmtree(noface_dir)
    os.mkdir(noface_dir)

    progress_count = 0
    progress_current = 0
    progress_count = len(files_list)
    print("number of images: {}".format(progress_count))

    for fpath in files_list:
        print("\rprogress: {:02}%".format(int(progress_current / progress_count * 100)), end = " ")
        try:
            wp = WPicture(fpath)
        except:
            progress_current += 1
            continue
        img = misc.imread(wp.fpath)
        if img.ndim<2:
            progress_current += 1
            continue
        img = misc.imresize(img, np.dot(img.shape,  np.min((1024 / np.max(img.shape), 1)))[0:2].astype(int), interp='bilinear')
        img = ndimage.rotate(img, wp.rotate)
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
        wp.img = img
        wp.image_size = np.asarray(img.shape)[0:2]

        bounding_boxes, _ = align.detect_face.detect_face(wp.img, minsize, pnet, rnet, onet, threshold, factor)
        le_threshold_posi = np.where(bounding_boxes[:, -1] < w_THRESHOLD_SCORE_)
        bounding_boxes = np.delete(bounding_boxes, le_threshold_posi, 0)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces <= 0:
            shutil.copy(wp.fpath, noface_dir)
            progress_current += 1
            continue

        absposition, relposition = calc_abs_rel_position(bounding_boxes, wp.image_size)
        comp_threshold_result = np.zeros([bounding_boxes.shape[0], 2], dtype=np.int)
        comp_threshold_result[:,0] = np.maximum(relposition[:,2] - relposition[:,0] - w_HEIGHT_WIDTH_, 0)
        comp_threshold_result[:,1] = np.maximum(relposition[:,3] - relposition[:,1] - w_HEIGHT_WIDTH_, 0)
        le_threshold_posi = np.where(np.min(comp_threshold_result, axis = 1) == 0)
        bounding_boxes = np.delete(bounding_boxes, le_threshold_posi, 0)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces <= 0:
            shutil.copy(wp.fpath, noface_dir)
            progress_current += 1
            continue

        absposition, relposition = calc_abs_rel_position(bounding_boxes, wp.image_size)
        wp.setBoundingBox(bounding_boxes)
        wp.setFaceAbsPosition(absposition)
        wp.setFaceRelPosition(relposition)

        ret_wp_list.append(wp)
        progress_current += 1

    print("\rprogress: 100%")

    return ret_wp_list


def getSimilar(wp_list, args):
    ret_group_list = []
    tree = bktree.BKTree(compareDHash, wp_list)
    record = {}
    SimilarResult = []

    def makeResult(bt, obj):
        if obj in record:
            return []
        record[obj] = True
        res = [obj]
        neri = bt.query(obj, w_SIMILAR_)
        for i in range(1, len(neri)):
            res = res + makeResult(bt, neri[i][1])
        return res

    for wp in wp_list:
        if wp in record:
            continue
        SimilarResult.append(makeResult(tree, wp))
    for g in SimilarResult:
        ret_group_list.append(WPictureGroup(g))

    return ret_group_list

def getFeatureByTorch(wp_group_list, args):
    dst_dir = os.path.join(args.input_dir, "align_face")
    if os.path.exists(dst_dir):
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        else:
            os.remove(dst_dir)
    os.mkdir(dst_dir)

    releation_dict = {}
    for idx, group in enumerate(wp_group_list):
        releation_dict[idx] = group
        delegate = group.getGroupDelegate()
        absposition = delegate.getFaceAbsPosition()
        img = delegate.img

        for i in range(absposition.shape[0]):
            det = absposition[i, :]

            cropped = img[det[1]:det[3],det[0]:det[2],:]
            scaled = misc.imresize(cropped, (w_TORCH_IMAGE_SIZE_, w_TORCH_IMAGE_SIZE_), interp='bilinear')
            output_filename = os.path.join(dst_dir, "{}__{}.png".format(idx, i))
            misc.imsave(output_filename, scaled)


    os.system("th feature.lua {}".format(args.input_dir))

    fnames = np.genfromtxt(os.path.join(args.input_dir, "labels.csv"), dtype="U", delimiter=",")
    ffeature = np.genfromtxt(os.path.join(args.input_dir, "reps.csv"), delimiter = ",")

    for idx, n in enumerate(fnames):
        name = n.split('.')[0]
        k_nth = name.split('__') 
        k = k_nth[0]
        nth = k_nth[1] 

        group = releation_dict[int(k)]
        if group.embeddings is None:
            group.embeddings = np.zeros([group.getGroupDelegate().getFaceCount(), 128], dtype=np.float)

        print(ffeature.shape)
        print(nth, idx)
        group.embeddings[int(nth), :] = ffeature[idx, :]

def getFeatureByTensorflow(wp_group_list, args):
    network = importlib.import_module("models.inception_resnet_v1", 'inference')
    with tf.Graph().as_default():
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
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
            # Only update statistics during training mode
            'is_training': phase_train_placeholder
        }
        # Build the inference graph
        prelogits, _ = network.inference(images_placeholder, 1, phase_train=phase_train_placeholder, weight_decay = 1.0)
        pre_embeddings = slim.fully_connected(prelogits, 128, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.0),
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                scope='Bottleneck', reuse=False)
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3, allow_empty=True)
        #saver = tf.train.Saver(allow_empty=True)

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:False})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:False})

            # saver.restore(sess, "/home/chen/demo_dir/facenet_tensorflow_train/trained_model_2017_05_17_11_08_resnet/20170518-133727/model-20170518-133727.ckpt-39024")
            saver.restore(sess, "/home/chen/demo_dir/facenet_tensorflow_train/trained_model_2017_05_17_11_08_resnet/20170518-133727/model-20170518-133727.ckpt-1000060")

            progress_count = 0
            progress_current = 0
            progress_count = len(wp_group_list)
            print("number of groups: {}".format(progress_count))

            for group in wp_group_list:
                print("\rprogress: {:02}%".format(int(progress_current / progress_count * 100)), end = " ")
                delegate = group.getGroupDelegate()
                nrof_samples = delegate.getFaceCount()
                image_size = images_placeholder.get_shape()[1]
                images = np.zeros((nrof_samples, image_size, image_size, 3))
                for nth in range(nrof_samples):
                    det = delegate.getFaceAbsPosition()[nth, :]
                    cropped = delegate.img[int(det[1]):int(det[3]),int(det[0]):int(det[2]),:]
                    img = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    img = facenet.prewhiten(img)
                    img = facenet.crop(img, False, image_size)
                    img = facenet.flip(img, False)
                    images[nth,:,:,:] = img

                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder:images, phase_train_placeholder:False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                group.setFeatureForDelegate(emb_array)
                progress_current += 1
            print("\rprogress: 100%")

def getFeature(wp_group_list, args):
    if not args.torch:
        getFeatureByTensorflow(wp_group_list, args)
    else:
        getFeatureByTorch(wp_group_list, args)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory with images.')
    parser.add_argument('--load_json', type=str, help='load json file makes wp_list')
    parser.add_argument('--torch', action='store_true', help='use torch get embeddings', default=False)

    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    if not os.path.isdir(args.input_dir):
        print("{} is not a directory".format(args.input_dir))
        exit(-1)

    wp_list = None

    if args.load_json is None:
        images = []
        images += glob.iglob(os.path.join(args.input_dir, "*.jpg"))
        images += glob.iglob(os.path.join(args.input_dir, "*.png"))
        wp_list = detectFace(images, args)
        with open(os.path.join(args.input_dir, "wplist.json"), "w+") as fwrite:
            fwrite.write(WPictureWriter(wp_list))
    else:
        wp_list = []
        with open(os.path.join(args.input_dir, "wplist.json"), "r") as fread:
            wp_json = json.load(fread)
            for w_json in wp_json:
                wp = WPictureReader(w_json)
                wp_list.append(wp)

    wp_group_list = getSimilar(wp_list, args)
    getFeature(wp_group_list, args)

    gallery = []
    bbox = []
    signatures = []
    for group in wp_group_list:
        delegate = group.getGroupDelegate()
        for i in range(group.embeddings.shape[0]):
            delegate = group.getGroupDelegate()
            g_list = []
            for g in group:
                g_list.append(os.path.basename(g.fpath).split('.')[0])
            gallery.append(g_list)
            b_list = [group.delegate, i]
            bbox.append(b_list + delegate.getFaceRelPosition()[i, :].tolist())
            signatures.append(group.embeddings[i, :].tolist())

    gdir = os.path.join(args.input_dir, "gallery.npy")
    pdir = os.path.join(args.input_dir, "position.npy")
    sdir = os.path.join(args.input_dir, "signatures.npy")

    if os.path.exists(gdir):
        os.remove(gdir)
    if os.path.exists(pdir):
        os.remove(pdir)
    if os.path.exists(sdir):
        os.remove(sdir)

    np.save(gdir, np.array(gallery))
    np.save(pdir, np.array(bbox))
    np.save(sdir, np.array(signatures))
