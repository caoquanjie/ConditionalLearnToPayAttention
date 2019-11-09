from utils import *
import tensorflow as tf



def model(config,images,labels,is_train):
    """ Build the VGG16 net. """
    print("Building the CNN...")

    conv1_1 = ConvBNReLU(images, w_shape=[3, 3, 3, 64], b_shape=[64, ], axis=-1, phase=is_train, name='conv1_1')
    conv1_1 = dropout_layer(conv1_1,drop_rate=config.conv_drop_rate_03,is_train=is_train)
    conv1_2 = ConvBNReLU(conv1_1, w_shape=[3, 3, 64, 64], b_shape=[64, ], axis=-1, phase=is_train, name='conv1_2')
    #pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = ConvBNReLU(conv1_2, w_shape=[3, 3, 64, 128], b_shape=[128, ], axis=-1, phase=is_train, name='conv2_1')
    conv2_1 = dropout_layer(conv2_1,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv2_2 = ConvBNReLU(conv2_1, w_shape=[3, 3, 128, 128], b_shape=[128, ], axis=-1, phase=is_train, name='conv2_2')
    #pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = ConvBNReLU(conv2_2, w_shape=[3, 3, 128, 256], b_shape=[256, ], axis=-1, phase=is_train, name='conv3_1')
    conv3_1 = dropout_layer(conv3_1,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv3_2 = ConvBNReLU(conv3_1, w_shape=[3, 3, 256, 256], b_shape=[256, ], axis=-1, phase=is_train, name='conv3_2')
    conv3_2 = dropout_layer(conv3_2,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv3_3 = ConvBNReLU(conv3_2, w_shape=[3, 3, 256, 256], b_shape=[256, ], axis=-1, phase=is_train, name='conv3_3')
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = ConvBNReLU(pool3, w_shape=[3, 3, 256, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv4_1')
    conv4_1 = dropout_layer(conv4_1,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv4_2 = ConvBNReLU(conv4_1, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv4_2')
    conv4_2 = dropout_layer(conv4_2,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv4_3 = ConvBNReLU(conv4_2, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv4_3')
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = ConvBNReLU(pool4, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv5_1')
    conv5_1 = dropout_layer(conv5_1,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv5_2 = ConvBNReLU(conv5_1, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv5_2')
    conv5_2 = dropout_layer(conv5_2,drop_rate=config.conv_drop_rate_04,is_train=is_train)
    conv5_3 = ConvBNReLU(conv5_2, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv5_3')
    pool5 = max_pool(conv5_3, 'pool5')

    conv6_1 = ConvBNReLU(pool5, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv6_1')
    pool6 = max_pool(conv6_1, 'pool6')

    conv7_1 = ConvBNReLU(pool6, w_shape=[3, 3, 512, 512], b_shape=[512, ], axis=-1, phase=is_train, name='conv7_1')
    pool7 = max_pool(conv7_1, 'pool7')

    fc6 = fc_layer(pool7, "fc6", w_shape=[2048, 512], b_shape=[512])
    relu6 = tf.nn.relu(fc6)
    relu6 = dropout_layer(relu6,drop_rate=config.fc_drop_rate,is_train=is_train)

    g_init = relu6


    #build the RNN network
    print("Building the RNN...")

    lstm_drop_rate = tf.cond(is_train,lambda :0.3,lambda :0.0)
    # Setup the LSTM
    lstm = tf.nn.rnn_cell.LSTMCell(
        config.num_lstm_units,
        initializer=tf.random_uniform_initializer(
            minval=-config.fc_kernel_initializer_scale,
            maxval=config.fc_kernel_initializer_scale))

    lstm = tf.nn.rnn_cell.DropoutWrapper(
        lstm,
        input_keep_prob=1.0 - lstm_drop_rate,
        output_keep_prob=1.0 - lstm_drop_rate,
        state_keep_prob=1.0 - lstm_drop_rate)



    # Initialize the LSTM using the global feature
    with tf.variable_scope("initialize"):
        initial_memory, initial_output = initialize(config,g_init,is_train=is_train)
        initial_state = initial_memory, initial_output

    # Prepare to run
    probs = []
    predictions = []
    cross_entropies = []
    feats3_pictures = []
    num_steps = config.label_length-1
    last_output = initial_output
    last_memory = initial_memory
    last_state = last_memory,last_output
    global_feats = g_init

    REUSE = None
    for idx in range(num_steps):

        ga_3, init_p3,pic = compatibility_func(conv5_3, global_feats)

        with tf.variable_scope("lstm",reuse=REUSE):
            current_input = ga_3
            output, state = lstm(current_input,last_state)
            memory, _ = state

        with tf.variable_scope("decoder"):
            expanded_output = ga_3
            logits = decode(config, expanded_output)  # (batch,11)
            prob = tf.nn.softmax(logits)
            prediction = tf.cast(tf.argmax(prob, 1), tf.int64)
            probs.append(tf.reduce_max(prob, 1))
            predictions.append(prediction)


        # Compute the loss for this step, if necessary
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = labels[:,idx],
            logits = logits))
        cross_entropies.append(cross_entropy)
        feats3_pictures.append(init_p3)
        last_state = state
        global_feats = output

        REUSE=True


    # Compute the final loss, if necessary
    var_list = tf.trainable_variables()
    cross_entropy_loss = tf.add_n(cross_entropies)

    reg_loss= tf.add_n([tf.nn.l2_loss(v) for v in var_list
                       if 'bias' not in v.name]) * 5e-4

    total_loss = cross_entropy_loss + reg_loss
    accuracy = calculate_acc(predictions,labels[:,:5])



    print("RNN built.")
    global_step = tf.Variable(0,trainable=False)
    learning_rate_start = config.initial_learning_rate
    learning_rate_per_decay = config.learning_rate_per_decay
    num_step_per_decay = config.num_step_per_decay

    learning_rate = tf.train.exponential_decay(
        learning_rate_start,
        global_step,
        num_step_per_decay,
        learning_rate_per_decay,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-4)

    with tf.name_scope("metrics"):
        tf.summary.image("image",images)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("learning rate",learning_rate)

    var_list_lstm = [var for var in var_list
                    if 'lstm/lstm_cell' in var.name]
    var_list_else = [var for var in var_list
                    if 'lstm/lstm_cell' not in var.name]
    for var in var_list_lstm:
        tf.summary.histogram(var.op.name, var)

    with tf.name_scope("train"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(total_loss,global_step=global_step)


    summary = tf.summary.merge_all()

    return images,feats3_pictures,total_loss,cross_entropy_loss,reg_loss,accuracy,train_op,probs,predictions,learning_rate,summary,global_step



def initialize(config,global_feature,is_train):

    temp1 = fc_layer(global_feature,"fc_a1",
                    w_shape=[global_feature.shape[1],config.dim_initalize_layer],
                    b_shape=[config.dim_initalize_layer])
    temp1 = tf.nn.tanh(temp1)

    temp1 = dropout_layer(temp1,config.fc_drop_rate,is_train=is_train)

    memory = fc_layer(temp1,"fc_a2",
                    w_shape=[temp1.shape[1],config.num_lstm_units],
                    b_shape=[config.num_lstm_units])

    temp2 = fc_layer(global_feature, "fc_b1",
                     w_shape=[global_feature.shape[1], config.dim_initalize_layer],
                     b_shape=[config.dim_initalize_layer])
    temp2 = tf.nn.tanh(temp2)
    temp2 = dropout_layer(temp2,drop_rate=config.fc_drop_rate,is_train=is_train)

    output = fc_layer(temp2,"fc_b2",
                    w_shape=[temp2.shape[1],config.num_lstm_units],
                    b_shape=[config.num_lstm_units])

    return memory, output


def decode(config,expanded_output):
    logits = decoder_layer(expanded_output, "cls",
                           w_shape=[expanded_output.get_shape().as_list()[1], config.num_classes],
                           b_shape=[config.num_classes, ])
    return logits


def calculate_acc(predictions,labels):
    masks = tf.to_float(tf.not_equal(labels, 10))
    predictions_correct = []
    for i in range(5):
        ground_truth = tf.cast(labels[:, i], tf.int64)
        prediction_correct = tf.where(
            tf.equal(predictions[i], ground_truth),
            tf.cast(masks[:, i], tf.float32),
            tf.cast(tf.zeros_like(predictions[i]), tf.float32))
        predictions_correct.append(prediction_correct)

    predictions_correct = tf.stack(predictions_correct, axis=1)
    accuracy = tf.reduce_sum(predictions_correct) \
               / tf.reduce_sum(masks)

    return accuracy










