import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def read_tf(filename,flag):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'length': tf.FixedLenFeature([], tf.int64),
                                           'digits': tf.FixedLenFeature([5], tf.int64)
                                       })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.cast(image,tf.float32)

    if flag:
        image = tf.random_crop(image, [54, 54, 3])
        #image = tf.clip_by_value(image,0,1)
        #image = tf.image.random_flip_left_right(image)
        #image = tf.image.resize_image_with_crop_or_pad(image,64,64)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,54,54)

    length = features['length']
    digits = features['digits']
    return image, digits

def generate_batch(
        example,
        #min_queue_examples,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1
    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=20 * batch_size,
            min_after_dequeue=10 * batch_size)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            capacity=20 * batch_size)

    return ret


# with tf.Session() as sess:
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     image, label = read_tf('/mnt/data/cqj/tfrecords_dataset/crop_data_aug/val.tfrecords',False)
#     images, labels = generate_batch([image, label], 30, shuffle=True)
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(20):
#         example, l = sess.run([images, labels])
#         print(i, l)
#
#         plt.title(l[i])
#         plt.imshow(example[i])
#         plt.show()
#     coord.request_stop()
#     coord.join(threads)
#     sess.close()
