import tensorflow as tf
from model_utils import fc_layer,compatibility_func
from base_model import BaseModel

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn == 'vgg16':
            self.build_vgg16()

        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """


        print("Building the CNN...")

        config = self.config
        # set the placeholders
        images = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.images_size, config.images_size, 3])



        conv1_1_feats = self.nn.conv2d(images, 64, name='conv1_1')
        conv1_1_feats = self.nn.batch_norm(conv1_1_feats)
        conv1_1_feats = tf.nn.relu(conv1_1_feats)
        conv1_1_feats = tf.layers.dropout(conv1_1_feats, rate=0.3, training=self.is_train)
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name='conv1_2')
        conv1_2_feats = self.nn.batch_norm(conv1_2_feats)
        conv1_2_feats = tf.nn.relu(conv1_2_feats)
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name='pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name='conv2_1')
        conv2_1_feats = self.nn.batch_norm(conv2_1_feats)
        conv2_1_feats = tf.nn.relu(conv2_1_feats)
        conv2_1_feats = tf.layers.dropout(conv2_1_feats, rate=0.4, training=self.is_train)
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name='conv2_2')
        conv2_2_feats = self.nn.batch_norm(conv2_2_feats)
        conv2_2_feats = tf.nn.relu(conv2_2_feats)
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name='pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name='conv3_1')
        conv3_1_feats = self.nn.batch_norm(conv3_1_feats)
        conv3_1_feats = tf.nn.relu(conv3_1_feats)
        conv3_1_feats = tf.layers.dropout(conv3_1_feats, rate=0.4, training=self.is_train)
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name='conv3_2')
        conv3_2_feats = self.nn.batch_norm(conv3_2_feats)
        conv3_2_feats = tf.nn.relu(conv3_2_feats)
        conv3_2_feats = tf.layers.dropout(conv3_2_feats, rate=0.4, training=self.is_train)
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name='conv3_3')
        conv3_3_feats = self.nn.batch_norm(conv3_3_feats)
        conv3_3_feats = tf.nn.relu(conv3_3_feats)
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name='pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name='conv4_1')
        conv4_1_feats = self.nn.batch_norm(conv4_1_feats)
        conv4_1_feats = tf.nn.relu(conv4_1_feats)
        conv4_1_feats = tf.layers.dropout(conv4_1_feats, rate=0.4, training=self.is_train)
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name='conv4_2')
        conv4_2_feats = self.nn.batch_norm(conv4_2_feats)
        conv4_2_feats = tf.nn.relu(conv4_2_feats)
        conv4_2_feats = tf.layers.dropout(conv4_2_feats, rate=0.4, training=self.is_train)
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name='conv4_3')
        conv4_3_feats = self.nn.batch_norm(conv4_3_feats)
        conv4_3_feats = tf.nn.relu(conv4_3_feats)
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name='pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name='conv5_1')
        conv5_1_feats = self.nn.batch_norm(conv5_1_feats)
        conv5_1_feats = tf.nn.relu(conv5_1_feats)
        conv5_1_feats = tf.layers.dropout(conv5_1_feats, rate=0.4, training=self.is_train)
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name='conv5_2')
        conv5_2_feats = self.nn.batch_norm(conv5_2_feats)
        conv5_2_feats = tf.nn.relu(conv5_2_feats)
        conv5_2_feats = tf.layers.dropout(conv5_2_feats, rate=0.4, training=self.is_train)
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name='conv5_3')
        conv5_3_feats = self.nn.batch_norm(conv5_3_feats)
        conv5_3 = tf.nn.relu(conv5_3_feats)
        pool5_feats = self.nn.max_pool2d(conv5_3, name='pool5')

        fc6 = self.nn.dropout(pool5_feats)
        fc6 = fc_layer(fc6, "fc6", w_shape=[25088, 512], b_shape=[512])
        relu6 = tf.nn.relu(fc6)

        self.g_init = relu6
        self.conv5_3 = conv5_3
        self.images = images



    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            global_feats = self.g_init
            conv5_3 = self.conv5_3
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])
        else:
            global_feats = self.g_init
            conv5_3 = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size,14,14,512])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

            last_memory2 = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output2 = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)


        lstm2 = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer=self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm2,
                input_keep_prob=1.0 - config.lstm_drop_rate,
                output_keep_prob=1.0 - config.lstm_drop_rate,
                state_keep_prob=1.0 - config.lstm_drop_rate)
        lstm3 = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer=self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm3,
                input_keep_prob=1.0 - config.lstm_drop_rate,
                output_keep_prob=1.0 - config.lstm_drop_rate,
                state_keep_prob=1.0 - config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            initial_memory, initial_output = self.initialize(global_feats)
            initial_state = initial_memory, initial_output
            initial_memory2 = tf.zeros([config.batch_size, config.num_lstm_units])
            initial_output2 = tf.zeros([config.batch_size, config.num_lstm_units])
            initial_memory3 = tf.zeros([config.batch_size, config.num_lstm_units])
            initial_output3 = tf.zeros([config.batch_size, config.num_lstm_units])
        # Prepare to run

        inputs_sequences = []
        outputs_b = []
        gas = []
        word_embeds = []
        outputs = []
        predictions = []
        predictions2 = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            cross_entropies2 = []
            predictions_correct2 = []
            num_steps = config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_memory2 = initial_memory2
            last_output2 = initial_output2
            last_memory3 = initial_memory3
            last_output3 = initial_output3
            last_word = tf.zeros([config.batch_size], tf.int32)

        else:
            num_steps = 1
        last_state = last_memory, last_output
        last_state2 = last_memory2,last_output2
        last_memory3,last_output3 = tf.zeros([config.batch_size, config.num_lstm_units]),tf.zeros([config.batch_size, config.num_lstm_units])
        last_state3 = last_memory3,last_output3
        # Generate the words one by one
        REUSE = None
        for idx in range(num_steps):

            ga_3, init_p3,alpha = compatibility_func(conv5_3, last_output)

            if self.is_train:
                tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                      [1, 196])
                masked_alpha = alpha * tiled_masks
                alphas.append(tf.reshape(masked_alpha, [-1]))

            gas.append(ga_3)


            with tf.variable_scope("word_embedding", reuse=REUSE):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
                word_embeds.append(word_embed)

            with tf.variable_scope("lstm", reuse=REUSE):
                current_input = tf.concat([last_output2,ga_3, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            with tf.variable_scope("lstm2", reuse=REUSE):
                cur_ga,cur_p,cur_alpha = compatibility_func(conv5_3,output)
                cur_input = tf.concat([output,cur_ga,word_embed],1)
                output2, state2 = lstm2(cur_input,last_state2)
                memory2,_ = state2
                outputs_b.append(output2)
                inputs_sequences.append(cur_input)
            # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder",reuse=REUSE):
                expanded_output = tf.concat([ga_3,last_output2,word_embed],1)
                logits = self.decode(expanded_output)  # (batch,11)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction[0])


                # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sentences[:, idx],
                    logits=logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx]
                last_output2 = output2
                last_memory2 = memory2
                last_state2 = state2
                global_feats = output

            REUSE = True


        REUSE=None

        for idx in range(num_steps):
            input = inputs_sequences[::-1][idx]

            with tf.variable_scope("lstm3", reuse=REUSE):
                output3, state3 = lstm3(input, last_state3)
                memory3, _ = state3

                outputs.append(output3)

                last_state3 = state3
            REUSE = True

        REUSE=None

        for idx in range(num_steps):

            # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder2", reuse=REUSE):
                expanded_output = tf.concat([gas[idx], last_output2,last_output3, word_embeds[idx]], 1)
                logits2 = self.decode(expanded_output)  # (batch,11)
                probs2 = tf.nn.softmax(logits2)
                prediction2 = tf.argmax(logits2, 1)
                predictions2.append(prediction2[0])

            if self.is_train:
                cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sentences[:, idx],
                    logits=logits2)
                masked_cross_entropy = cross_entropy2 * masks[:, idx]
                cross_entropies2.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction2, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction2), tf.float32))
                predictions_correct2.append(prediction_correct)

                last_output3 = outputs[::-1][idx]
                last_output2 = outputs_b[idx]

            REUSE = True



        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis=1)
            alphas = tf.reshape(alphas, [config.batch_size, 196, -1])
            attentions = tf.reduce_sum(alphas, axis=2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * 196)

            reg_loss = tf.losses.get_regularization_loss()

            cross_entropies2 = tf.stack(cross_entropies2, axis=1)
            cross_entropy_loss2 = tf.reduce_sum(cross_entropies2) \
                                 / tf.reduce_sum(masks)

            total_loss = cross_entropy_loss  + reg_loss + attention_loss + cross_entropy_loss2

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.conv5_feats = conv5_3
        if self.is_train:
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            #self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            #self.attentions = attentions
            self.probs = probs
            self.predictions = predictions
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.initial_memory2 = initial_memory2
            self.initial_output2 = initial_output2
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_memory2 = last_memory2
            self.last_output2 = last_output2
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.memory2 = memory2
            self.output2 = output2
            self.probs = probs
            self.alpha = alpha

        print("RNN built.")
    #
    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config

        # use 2 fc layers to initialize
        temp1 = self.nn.dense(context_mean,
                              units = config.dim_initalize_layer,
                              activation = tf.tanh,
                              name = 'fc_a1')
        temp1 = self.nn.dropout(temp1)
        memory = self.nn.dense(temp1,
                               units = config.num_lstm_units,
                               activation = None,
                               name = 'fc_a2')

        temp2 = self.nn.dense(context_mean,
                              units = config.dim_initalize_layer,
                              activation = tf.tanh,
                              name = 'fc_b1')
        temp2 = self.nn.dropout(temp2)
        output = self.nn.dense(temp2,
                               units = config.num_lstm_units,
                               activation = None,
                               name = 'fc_b2')
        return memory, output


    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """

        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        temp = self.nn.dense(expanded_output,
                              units=config.dim_decode_layer,
                              activation=tf.tanh,
                              name='fc_1')
        temp = self.nn.dropout(temp)
        logits = self.nn.dense(temp,
                               units=config.vocabulary_size,
                               activation=None,
                               name='fc_2')
        return logits


    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)


        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


