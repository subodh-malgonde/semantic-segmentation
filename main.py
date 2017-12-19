import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import random
import time
from tqdm import *
import math
from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import argparse
import logging
from datetime import datetime

logging.basicConfig(filename='training.log',level=logging.INFO)


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

KEEP_PROB = 0.6
LEARNING_RATE = 0.00005
TRANSFER_LEARNING_MODE = False
CONTINUE_TRAINING = False

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, ["vgg16"], vgg_path)

    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    if TRANSFER_LEARNING_MODE:
        vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
        vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
        vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01)

    new_layer7_1x1_out = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                      name='new_layer7_1x1_out')
    new_layer7_1x1_out = tf.Print(new_layer7_1x1_out, [tf.shape(new_layer7_1x1_out)], message="Layer 7 shape before: ", first_n=1)

    new_layer7_1x1_upsampled = tf.layers.conv2d_transpose(new_layer7_1x1_out, filters=num_classes, kernel_size=(4, 4),
                                                          strides=(4, 4), name='new_layer7_1x1_out_upsampled')
    new_layer7_1x1_upsampled = tf.Print(new_layer7_1x1_upsampled, [tf.shape(new_layer7_1x1_upsampled)],
                                        message="Layer 7 shape after: ", first_n=1)

    new_layer4_1x1_out = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                      name="new_layer4_1x1_out")
    new_layer4_1x1_out = tf.Print(new_layer4_1x1_out, [tf.shape(new_layer4_1x1_out)], message="Layer 4 shape before: ", first_n=1)

    new_layer4_1x1_upsampled = tf.layers.conv2d_transpose(new_layer4_1x1_out, filters=num_classes, kernel_size=(3, 3),
                                                      strides=(2, 2), name="new_layer4_1x1_upsampled", padding='same')
    new_layer4_1x1_upsampled = tf.Print(new_layer4_1x1_upsampled, [tf.shape(new_layer4_1x1_upsampled)], message="Layer 4 shape after: ",
                                    first_n=1)

    new_layer3_1x1_out = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                      name="new_layer3_1x1_out")

    new_layer3_1x1_out = tf.Print(new_layer3_1x1_out, [tf.shape(new_layer3_1x1_out)], message="Layer 3 shape: ", first_n=1)

    out = tf.add(new_layer7_1x1_upsampled, new_layer4_1x1_upsampled)
    out = tf.add(out, new_layer3_1x1_out)

    new_final_layer_upsampled_4x = tf.layers.conv2d_transpose(out, filters=num_classes, kernel_size=(4, 4),
                                                      strides=(4, 4), name="new_final_layer_upsampled_4x")

    new_final_layer_upsampled_4x = tf.Print(new_final_layer_upsampled_4x, [tf.shape(new_final_layer_upsampled_4x)],
                                        message="final_layer_upsampled_4x  shape: ", first_n=1)

    new_final_layer_upsampled_8x = tf.layers.conv2d_transpose(new_final_layer_upsampled_4x, filters=num_classes, kernel_size=(5, 5),
                                                       strides=(2, 2), name="new_final_layer_upsampled_8x", padding='same')

    new_final_layer_upsampled_8x = tf.Print(new_final_layer_upsampled_8x, [tf.shape(new_final_layer_upsampled_8x)],
                                        message="final_layer_upsampled_8x  shape: ", first_n=1)

    return new_final_layer_upsampled_8x

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # correct_label = tf.reshape(correct_label, (-1, num_classes))
    logits = nn_last_layer

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label),
                                        name="cross_entropy")

    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    if TRANSFER_LEARNING_MODE:

        trainable_variables = []
        for variable in tf.trainable_variables():
            if "new_" in variable.name:
                trainable_variables.append(variable)
        training_op = opt.minimize(cross_entropy_loss, var_list=trainable_variables, name="training_op")
    else:
        training_op = opt.minimize(cross_entropy_loss, name="training_op")

    return logits, training_op, cross_entropy_loss

# tests.test_optimize(optimize)


def train_nn(sess, epochs, data_folder, image_shape, batch_size, training_image_paths, validation_image_paths, train_op,
             cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    # Create function to get batches
    get_batches_fn_training = helper.gen_batch_function(data_folder, image_shape, training_image_paths)

    training_batch_generator = get_batches_fn_training(batch_size)

    samples_per_epoch = len(training_image_paths)
    batches_per_epoch = math.floor(samples_per_epoch/batch_size)

    print("Actual learning rate is", LEARNING_RATE)
    print("Keep probability is", KEEP_PROB)

    for epoch in range(epochs):
        for batch in tqdm(range(batches_per_epoch)):
            X_batch , y_batch = next(training_batch_generator)
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE
            })
        validation_loss = evaluate(validation_image_paths, data_folder, image_shape, sess, input_image, correct_label,
                                   keep_prob, cross_entropy_loss)
        training_loss = evaluate(training_image_paths, data_folder, image_shape, sess, input_image, correct_label,
                                   keep_prob, cross_entropy_loss)
        print("Epoch %d:" % (epoch + 1), "Training loss: %.4f," % training_loss, "Validation loss: %.4f" % validation_loss)
        logging.info("Epoch %d: Training loss: %.4f,  Validation loss: %.4f" % (epoch + 1, training_loss, validation_loss))

    print("Saving the model")
    if "saved_model" in os.listdir(os.getcwd()):
        shutil.rmtree("./saved_model")
    builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
    builder.add_meta_graph_and_variables(sess, ["vgg16"])
    builder.save()


def evaluate(image_paths, data_folder, image_shape, sess, input_image,correct_label, keep_prob, loss_op):
    data_generator_function = helper.gen_batch_function(data_folder, image_shape, image_paths)
    batch_size = 8
    data_generator = data_generator_function(batch_size)
    num_examples = int(math.floor(len(image_paths)/batch_size)*batch_size)
    total_loss = 0
    for offset in range(0, num_examples, batch_size):
        X_batch, y_batch = next(data_generator)
        loss = sess.run([loss_op], feed_dict={input_image: X_batch, correct_label: y_batch, keep_prob: 1.0})
        total_loss += (loss[0] * X_batch.shape[0])
    return total_loss/num_examples


def run():
    global LEARNING_RATE
    global KEEP_PROB
    global TRANSFER_LEARNING_MODE
    global CONTINUE_TRAINING

    logging.info('------------------- START ------------------------')
    logging.info('%s: Training begins' % datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        nargs='?',
        default=1,
        help='Number of epochs.'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        nargs='?',
        default=0.0001,
        help='Learning rate'
    )

    parser.add_argument(
        '-k',
        '--keep_probability',
        type=float,
        nargs='?',
        default=1.0,
        help='Keep probability for dropout'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        nargs='?',
        default=16,
        help='Batch size.'
    )

    parser.add_argument("-t", "--test", help="Test mode on", action="store_true")
    parser.add_argument("-tlo", "--transfer_learn_off", help="Transfer learning mode off", action="store_true")
    parser.add_argument("-ct", "--continues_training", help="Continue from where you left off", action="store_true")

    args = parser.parse_args()

    num_epochs = args.num_epochs
    LEARNING_RATE = args.learning_rate
    KEEP_PROB = args.keep_probability
    batch_size = args.batch_size
    testing_mode = args.test
    CONTINUE_TRAINING = args.continues_training
    TRANSFER_LEARNING_MODE = False if args.transfer_learn_off else True

    print("Number of epochs:", num_epochs)
    print("learning rate:", LEARNING_RATE)
    print("Keep prob:", KEEP_PROB)
    print("Batch size:", batch_size)
    print("Training mode:", "OFF" if testing_mode else "ON")
    print("Trasfer learning mode:", "ON" if TRANSFER_LEARNING_MODE else "OFF")
    print("Continue training?:", "YES" if CONTINUE_TRAINING else "NO")

    logging.info('Num epochs: %d, learning rate: %.6f, keep prob: %.2f, batch size: %d' % (num_epochs, LEARNING_RATE, KEEP_PROB, batch_size))


    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        if not testing_mode:
            # Path to vgg model
            data_folder = os.path.join(data_dir, 'data_road/training')
            image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))

            training_image_paths, validation_image_paths = train_test_split(image_paths, test_size=0.2)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            if CONTINUE_TRAINING:
                vgg_path = './saved_model'
            else:
                vgg_path = os.path.join(data_dir, 'vgg')

            #Build NN using load_vgg, layers, and optimize function
            vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor,\
            vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(sess, vgg_path)

            if CONTINUE_TRAINING:
                logits_operation_name = "new_final_layer_upsampled_8x/BiasAdd"
                graph = tf.get_default_graph()
                output_tensor = graph.get_operation_by_name(logits_operation_name).outputs[0]
                train_op = graph.get_operation_by_name("training_op")
                cross_entropy_loss = graph.get_operation_by_name("cross_entropy")
                correct_label = graph.get_tensor_by_name("correct_label:0")
                learning_rate = graph.get_tensor_by_name("learning_rate:0")
                print(cross_entropy_loss)
                print(correct_label)
                print(learning_rate)
            else:
                output_tensor = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor, num_classes)
                correct_label = tf.placeholder(tf.int8, (None,) + image_shape + (num_classes,), name="correct_label")
                learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
                output_tensor, train_op, cross_entropy_loss = optimize(output_tensor, correct_label, learning_rate, num_classes)

            if not CONTINUE_TRAINING:
                if TRANSFER_LEARNING_MODE:
                    my_variable_initializers = [var.initializer for var in tf.global_variables() if 'new_' in var.name]
                    sess.run(my_variable_initializers)
                else:
                    sess.run(tf.global_variables_initializer())

            #Train NN using the train_nn function
            train_nn(sess, epochs=num_epochs, data_folder=data_folder,image_shape=image_shape, batch_size=batch_size,
                     training_image_paths=training_image_paths, validation_image_paths=validation_image_paths,
                     train_op=train_op, cross_entropy_loss=cross_entropy_loss, input_image=vgg_input_tensor,
                     correct_label=correct_label, keep_prob=vgg_keep_prob_tensor, learning_rate=learning_rate)

        else:
            test_model()


        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

    logging.info('-------------------- END ----------------------')


def test_model():
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    with tf.Session() as sess:
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'

        logits_operation_name = "new_final_layer_upsampled_8x/BiasAdd"

        tf.saved_model.loader.load(sess, ["vgg16"], "./saved_model")

        graph = tf.get_default_graph()
        vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
        vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

        logits_tensor = graph.get_operation_by_name(logits_operation_name).outputs[0]
        helper.save_inference_samples(runs_dir=runs_dir, data_dir=data_dir, sess=sess,image_shape=image_shape,
                                      logits=logits_tensor, keep_prob=vgg_keep_prob_tensor, input_image=vgg_input_tensor)


if __name__ == '__main__':
    run()
