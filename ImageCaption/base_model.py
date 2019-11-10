import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import copy
import json
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN
import logging
import logging
import skimage
import skimage.transform
import skimage.io
import matplotlib.cm as cm
import re
from PIL import Image



# log info
def set_log_info():
    logger = logging.getLogger('result')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler('result.log')
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

logger = set_log_info()

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data,test_data,vocabulary,FLAGS):
        """ Train the model using the COCO train2014 data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for k in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks}
                _, summary, global_step,predict,acc = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step,
                                                    self.predictions,
                                                    self.accuracy],
                                                    feed_dict=feed_dict)
                if (global_step + 1) % 100 == 0:
                    logger.info('step {}: train_acc = {:3.4f}\tpredict = {}'.format(
                        global_step, acc, np.array(predict)))

                if (global_step + 1) % config.save_period == 0:
                    self.save()

                train_writer.add_summary(summary, global_step)
            train_data.reset()
        self.save()
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")

        if not os.path.exists('./visual'):
            os.mkdir('./visual')
        if not os.path.exists('./visual1'):
            os.mkdir('./visual1')
        config = self.config

        results = []
        print('config.eval_result_dir:', config.eval_result_dir)
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch = eval_data.next_batch()
            caption_data, pics = self.beam_search(sess, batch, vocabulary)
            print(len(pics))
            fake_cnt = 0 if k < eval_data.num_batches - 1 \
                else eval_data.fake_count
            for l in range(eval_data.batch_size - fake_cnt):

                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)

                if config.visual:
                    if not os.path.exists('./visual/sample{}_{}'.format(k, l)):
                        os.mkdir('./visual/sample{}_{}'.format(k, l))

                    if not os.path.exists('./visual1/sample{}_{}'.format(k, l)):
                        os.mkdir('./visual1/sample{}_{}'.format(k, l))
                    caption_list = caption.split()
                    terminate_0 = caption_list[-1].split('.')[0]
                    terminate_1 = list(caption_list[-1])[-1]

                    caption_list.pop()
                    caption_list.append(terminate_0)
                    caption_list.append(terminate_1)

                    print(caption_list)

                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    # img = plt.imread(image_file)
                    img = Image.open(image_file)
                    img = img.resize((224, 224))

                    plt.imshow(img)
                    plt.title(caption)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig('./visual/sample{}_{}/result.png'.format(k, l))
                    for a in range(len(caption_list)):
                        plt.imshow(img)
                        plt.title(caption_list[a])
                        im = skimage.transform.pyramid_expand(pics[a][l, :].reshape(14, 14), upscale=16, sigma=20)
                        plt.imshow(im, alpha=0.8)
                        plt.set_cmap(cm.Greys_r)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig('./visual/sample{}_{}'.format(k, l) + '/%d.png' % a)


                results.append({'image_id': eval_data.image_ids[idx].item(),
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                image_file = batch[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = plt.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.test_result_dir,
                                         image_name+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")


    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        conv5_3, initial_memory, initial_output,initial_memory2,initial_output2 = sess.run(
            [self.conv5_3, self.initial_memory, self.initial_output,self.initial_memory2,self.initial_output2],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       memory2 = initial_memory2[k],
                                       output2 = initial_output2[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        pics = []
        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_memory2 = np.array([pcl[b].memory2
                                        for pcl in partial_caption_data_lists],
                                       np.float32)
                last_output2 = np.array([pcl[b].output2
                                        for pcl in partial_caption_data_lists],
                                       np.float32)

                memory, output,memory2,output2, pic,scores = sess.run(
                    [self.memory, self.output, self.memory2,self.output2,self.alpha,self.probs],
                    feed_dict = {self.conv5_feats: conv5_3,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output,
                                 self.last_memory2: last_memory2,
                                 self.last_output2: last_output2,
                                 self.images : images})
                if b == 0:
                    pics.append(pic)

                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           memory2[k],
                                           output2[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results,pics

    def save(self):
        """ Save the model. """
        config = self.config
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path + ".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step) + ".npy")

        print("Loading the model from %s..." % save_path)
        data_dict = np.load(save_path,encoding="latin1").item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." % count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." % data_path)
        # import pdb; pdb.set_trace()
        data_path = data_path.strip()
        data_dict = np.load(os.getcwd() + '/' + data_path, encoding='latin1', allow_pickle=True).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." % count)


