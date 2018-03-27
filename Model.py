import tensorflow as tf

class Model():
    def __init__(self):
        self.build_model()

        # hyperparemeters
        self.dense1_input_size = 512
        self.dropout_rate = 0.05
        self.dense2_input_size = 512
        self.dense3_input_size = 512

        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.997

        self.alpha = 1
        self.min_alpha = 0.05
        self.alph_decay = 0.997

        self.gama = 0.99

        self.dense1_output_dim = 512

        # memory container
        self.memory = []

        self.update_slow_target_every = 100

    def calc_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.min_epsilon, epsilon)

    def calc_alpha(self):
        alpha = self.alpha * self.alpha_decay
        self.alpha = max(self.min_alpha, alpha)

    def build_model(self, state_dim_len, action_dim_len):
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, sate_dim_len])
        self.new_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim_len])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

    def build_q_networks(self):
        with tf.variable_scope('q_network') as scope:
            q_action_values_old = self.generate_network(True)
            q_action_values_new = self.generate_network(False)

    def build_slow_target_network(self):
        with tf.variable_scope('slow_target_network'):
            slow_target_action_values = tf.stop_gradient(
                    self.generate_network(
                        self.new_state,
                        is_trainable=False)
    def get_q_state(self):
        return sess.run(
                self.q_action_values_old,
                feed_dict={ state: old_state[None], is_training: False })

    def init_sess(self):
        self.sess = tf.Session()

    def build_network(self, is_trainable):
        self.dense1 = self.creat_dense_layer(
            self.state_dim_len, self.dense1_output_dim, 'dense1', is_trainable)

        self.dropout1 = self.crate_dropout_layer(
            self.dense1,'dropout1', is_trainable)

        self.dense2 = self.create_dense_layer(
            self.dropout1, self.dens2_output_dim, 'dense2', is_trainable)

        self.dropout2 = self.create_dropout_layer(
            self.dense2, 'dropout2', is_trainable)

        self.dense3 = self.create_dense_layer(
            self.dropout2, self.dens2_output_dim, 'dense2', is_trainable)

        self.droput3 = self.create_dropout_layer(
            self.dropout3, 'dropout3', is_trainable)

        self.activation_values = tf.squeeze(
                tf.layers.dense(
                    self.dropout3,
                    self.action_dim_len,
                    trainable=is_trainable,
                    name='dense3'))

    def create_dense_layer(self, input_dim, output_dim, name, is_trainable):
        return tf.layers.dense(
            input_dim,
            output_dim,
            activation=tf.nn.relu,
            trainable=is_trainable)

    def create_dropout_layer(self, input_dim, name, is_trainable):
        return tf.layers.dropout(
            input_dim,
            rate=self.droput_rate,
            training=is_trainable,
            name=name)
