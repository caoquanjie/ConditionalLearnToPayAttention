from utils import *
from config import Config
import tensorflow as tf
import logging
import os
from read_tf_records import generate_batch, read_tf
from Model import model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# log info
def set_log_info():
    logger = logging.getLogger('svhn')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler('svhn.log')
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger


#save and restore
def save_checkpoint(sess,step,saver,config):
    checkpoint_dir = config.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess=sess,
               save_path=checkpoint_dir+'model.ckpt',
               global_step=step)
    print('step %d,save model success'%step)

def load_checkpoint(sess,saver,config):
    checkpoint_dir = config.save_dir
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints and checkpoints.model_checkpoint_path:
        #checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
        #saver.restore(sess, os.path.join(checkpoint_dir,checkpoints_name))
        #saver.restore(sess,checkpoints.model_checkpoint_path)
        step = str(282002)
        saver.restore(sess,checkpoint_dir+"model.ckpt-"+step)
        print('step %d,load model success,contuinue training...'%int(step))
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('No checkpoint file found,initialize model... ')

logger = set_log_info()
config = Config()
is_train = tf.placeholder(dtype=tf.bool)


def input_data(dataname, batchsize, isShuffel, flag):
    image, label = read_tf(dataname, flag=flag)
    images, labels = generate_batch([image, label], batchsize, isShuffel)
    return images, labels

train_images, train_labels = input_data('crop_data_aug/train.tfrecords', batchsize=config.batch_size, isShuffel=False,
                                   flag=True)
test_images, test_labels = input_data('crop_data_aug/test.tfrecords', batchsize=config.batch_size,isShuffel=False,
                                 flag=False)

X_input = tf.cond(is_train, lambda: train_images, lambda: test_images)
Y_input = tf.cond(is_train, lambda: train_labels, lambda: test_labels)


images,feats2,feats3,total_loss,cross_entropy_loss,reg_loss,accuracy,train_op,probs,predictions,learning_rate,summary,global_step = \
    model(config,X_input,Y_input,is_train)




with tf.Session() as sess:
    writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=3,var_list=tf.global_variables())
    load_checkpoint(sess, saver,config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(config.num_step):

        if i and i % 10 == 0:

            Summary,step = sess.run([summary,global_step],feed_dict={is_train:True})
            writer.add_summary(Summary, step)


        result = sess.run([images, Y_input,feats2,feats3, total_loss, cross_entropy_loss, reg_loss, train_op, probs,predictions,
                               learning_rate, global_step, accuracy], \
                              feed_dict={is_train:True})

        if i and i % 100 == 0:

            logger.info(
                'step {}: total_loss = {:3.4f}\tlossXent = {:3.4f}\treg_loss = {:3.4f}'.format(
                    result[-2], result[4],result[5],result[6]))

            logger.info(
                'step {}: train_acc = {:3.4f}'.format(
                result[-2], result[-1]))

            logger.info(
                'step {}: learning rate = {:3.4f}'.format(result[-2],result[-3]))

        if i % 500 ==0:
            #print(result[1][0][0])
            save_p3_train_results(result[0], result[1],result[3],config, result[-2], result[-4], result[-5])
            save_p2_train_results(result[0], result[1], result[2],config, result[-2], result[-4], result[-5])

        if i and i % config.save_period == 0:
            save_checkpoint(sess,result[-2],saver,config)


        # eval result

        if i and i % 100 == 0:

            eval_num = 13000 // config.batch_size
            total_accuracy = 0

            for k in range(eval_num):
                result = sess.run([images,Y_input,feats2,feats3,probs,predictions,global_step,accuracy],
                                  feed_dict={is_train:False})
                total_accuracy += result[-1]

            acc = total_accuracy / eval_num


            logger.info(
                'step {}: Test_acc = {:3.4f}'.format(
                    result[-2], acc))

            save_p3_test_results(result[0], result[1], result[3], config, result[-2], result[-3], result[-4])
            save_p2_test_results(result[0], result[1], result[2], config, result[-2], result[-3], result[-4])
    save_checkpoint(sess, config.num_step, saver,config)