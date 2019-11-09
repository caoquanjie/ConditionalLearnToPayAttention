
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.image_size = 64
        self.label_length = 6
        self.num_lstm_units = 512
        self.dim_initalize_layer = 512
        self.num_classes = 11

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.conv_drop_rate_03 = 0.3
        self.conv_drop_rate_04 = 0.4
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3

        # about the optimization
        self.num_epochs = 200
        self.num_step = 300000
        self.batch_size = 32
        self.initial_learning_rate = 0.1
        self.learning_rate_per_decay = 0.5
        self.num_step_per_decay = 10000

        # about the saver
        self.save_period = 2000
        self.save_dir = './models/'
        self.summary_dir = './logs/'

        self.train_p3_result_dir = './train_p3_results/'
        self.train_p2_result_dir = './train_p2_results/'
        self.valid_result_dir = './valid_results'
        self.test_p3_result_dir = './test_p3_results/'
        self.test_p2_result_dir = './test_p2_results/'


