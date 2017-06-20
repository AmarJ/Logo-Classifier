#Amar Jasarbasic - Kaptur Technology
import tensorflow as tf
import numpy as np
import os
import sys
from scipy import ndimage
import re
import common
import scipy.misc
import PIL 
import time
from PIL import Image

imageWidth = 64
imageHeight = 32
imageCH = 3
dataSetDir = '../Kaptur-Data/datasets/training_sets/flickr_logos_27_dataset/flickr_logos_27_dataset_cropped_augmented_images'
pixDepth = 225.0

tfParams = tf.flags.FLAGS
tf.app.flags.DEFINE_integer("image_width", 64, "width")
tf.app.flags.DEFINE_integer("image_height", 32, "height")
tf.app.flags.DEFINE_integer("num_channels", 3, "# of channels")
tf.app.flags.DEFINE_integer("num_classes", 27, "# of logo classes")
tf.app.flags.DEFINE_integer("patch_size", 5,"patch size")

def model(data, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1,
          b_fc1, w_fc2, b_fc2):
    # First layer
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(data, w_conv1, [1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, w_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    # Third layer
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(h_pool2, w_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(
        h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, w_fc1) + b_fc1)

    # Output layer
    out = tf.matmul(h_fc1, w_fc2) + b_fc2

    return out


def loadInitialWeights(fileName):
    f = np.load(fileName)
    initialWeights = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    return initialWeights

def main():
    if len(sys.argv) > 1:
        test_image_fileName = sys.argv[1]
        if not os.path.exists(test_image_fileName):
            print("File: ", test_image_fileName, " does not exist.")
            sys.exit(-1)
    else:
        # Randomly selects an image from the test folder
        test_dirs = [os.path.join(dataSetDir, class_name, 'test')for class_name in common.CLASS_NAME]
        test_dir = np.random.choice(test_dirs)
        test_images_fileName = [test_image for test_image in os.listdir(test_dir)]
        test_image_fileName = np.random.choice(test_images_fileName, 1)[0]
        test_image_fileName = os.path.join(test_dir, test_image_fileName)
    
    print("Test image selected: ", test_image_fileName)
    img = Image.open(test_image_fileName)
    img = img.resize((imageWidth, imageHeight), PIL.Image.ANTIALIAS)
    img.save(test_image_fileName)

    test_image_org = (ndimage.imread(test_image_fileName).astype(np.float32) - pixDepth / 2) / pixDepth
    test_image_org.resize(imageHeight, imageWidth, imageCH)
    test_image = test_image_org.reshape((1, imageWidth, imageHeight, imageCH))
    
    #scipy.misc.imsave('test.png', test_image_org)

    # Training model
    graph = tf.Graph()
    with graph.as_default():
        w_conv1 = tf.Variable(tf.truncated_normal([tfParams.patch_size, tfParams.patch_size, tfParams.num_channels, 48],stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[48]))

        w_conv2 = tf.Variable(tf.truncated_normal([tfParams.patch_size, tfParams.patch_size, 48, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        w_conv3 = tf.Variable(tf.truncated_normal([tfParams.patch_size, tfParams.patch_size, 64, 128], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))

        w_fc1 = tf.Variable(tf.truncated_normal([16 * 4 * 128, 2048], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]))

        w_fc2 = tf.Variable(tf.truncated_normal([2048, tfParams.num_classes]))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[tfParams.num_classes]))

        params = [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2]

        # restore weights
        weightFile = "weights.npz"
        if os.path.exists(weightFile):
            initialWeights = loadInitialWeights(weightFile)
        else:
            initialWeights = None

        if initialWeights is not None:
            assert len(initialWeights) == len(params)
            assign_ops = [w.assign(v) for w, v in zip(params, initialWeights)]

        tf_test_image = tf.constant(test_image)
        logits = model(tf_test_image, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2)
        
        test_pred = tf.nn.softmax(logits)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        if initialWeights is not None:
            session.run(assign_ops)
            print('initialized by pre-learned weights')

        elif os.path.exists("../Kaptur-Data/models"):
            save_path = "../Kaptur-Data/models/deep_logo_model"
            saver.restore(session, save_path)

        else:
            print('initialized')
        pred = session.run([test_pred])
        print("Logo: ", common.CLASS_NAME[np.argmax(pred)])
        print("Probability: ", np.max(pred)*100.0)

if __name__ == '__main__':
    #execution time --start--
    start_time = time.time()
    main()
    print("Total execution time: %s seconds" % (time.time() - start_time))
