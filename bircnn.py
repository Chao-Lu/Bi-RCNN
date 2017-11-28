import tensorflow as tf
from config_environment import *
from config_hyper_parameter import MyConfig
from data_generator import *
import numpy as np
import time

l2_collection_name = "l2_collection"

def length2(sequence_batch):
    used = tf.sign(tf.abs(sequence_batch))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def length3(sequence_batch):
    used = tf.sign(tf.reduce_max(tf.abs(sequence_batch), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def cal_f1(prediction_label, actual_label):
    len1 = len(prediction_label)
    len2 = len(actual_label)
    assert len1 == len2
    TP = np.zeros(19)
    FP = np.zeros(19)
    FN = np.zeros(19)

    for i in range(0, len1):
        if actual_label[i] == prediction_label[i]:
            type_index = actual_label[i]
            TP[type_index] += 1
        else:
            type_index = actual_label[i]
            FN[type_index] += 1
        if prediction_label[i] != actual_label[i]:
            type_index = actual_label[i]
            FP[type_index] += 1
        else:
            pass
    P = TP + FN
    P_ = TP + FP
    precision = TP / P
    recall = TP / P_
    f1_score_list = 2 * precision * recall / (precision + recall)
    f1_score_mean = (sum(f1_score_list) - f1_score_list[9]) / 18
    return f1_score_list, f1_score_mean





    recal = np.zeros(18)
    precision = np.zeros(18)
    return

def get_prediction(hypothesis_f, hypothesis_b, alpha):
    hypothesis = alpha * hypothesis_f + (1-alpha) * hypothesis_b
    prdiction = tf.argmax(hypothesis, 1)
    return prdiction

def lstm_layer(input_sequence_batch, sequence_length, num_units, forget_bias, variable_scope):
    with tf.variable_scope(variable_scope, initializer=tf.orthogonal_initializer()):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                                      forget_bias=forget_bias,
                                                      state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                       inputs=input_sequence_batch,
                                       dtype=tf.float32,
                                       sequence_length=sequence_length)
        return outputs

def conv_layer(words_lstm_rst, rels_lstm_rst, concat_width,  conv_out_size, variable_scope):
    # lstm_rst : [batch_size, max_time, cell.output_size]
    # conv_input : [batch_size, max_time, concat_width, 1]
    # conv_shape : [1, concat_width, 1, conv_out_size]
    # bias_shape : [conv_out_size]
    # stride : [1,1,1,1]
    with tf.variable_scope(variable_scope):
        _conv_kernel = tf.get_variable(name='conv_kernel', shape=[1, concat_width, 1, conv_out_size], dtype=tf.float32, initializer=tf.contrib.keras.initializers.glorot_normal())
        _bias = tf.get_variable(name='bias', shape=[conv_out_size], dtype=tf.float32, initializer=tf.constant_initializer())
        tf.add_to_collection(l2_collection_name, _conv_kernel)
        tf.add_to_collection(l2_collection_name, _bias)

        words_lstm_rst_f = words_lstm_rst[:, :-1, :]
        words_lstm_rst_b = words_lstm_rst[:, 1:, :]
        conv_input = tf.concat([words_lstm_rst_f, rels_lstm_rst, words_lstm_rst_b], 2)
        conv_input_ = tf.stack([conv_input], 3) # [batch_size, max_time, concat_width, 1]

        # conv : [batch, max_time, 1, conv_out_size]
        conv = tf.nn.conv2d(input=conv_input_, filter=_conv_kernel, strides=[1, 1, conv_input_.get_shape()[2], 1], padding="VALID")
        bias = tf.nn.bias_add(conv, _bias)
        relu = tf.nn.relu(bias)
        return relu

def pool_layer(input_batch_data, config):
    # input_batch_data : [batch, max_time, 1, conv_out_size]
    pooling = tf.reduce_max(input_batch_data, 1)
    flatten = tf.reshape(pooling, [-1, config.conv_out_size])  # [batch, conv_out_size]
    return flatten

def softmax_layer(input_batch_data, input_size, output_size, variable_scope):
    with tf.variable_scope(variable_scope):
        w = tf.Variable(tf.random_normal([input_size, output_size]), name="weight", dtype=tf.float32)
        b = tf.Variable(tf.random_normal([output_size]), name="bias", dtype=tf.float32)
        tf.add_to_collection(l2_collection_name, w)
        tf.add_to_collection(l2_collection_name, b)

        logits = tf.matmul(input_batch_data, w) + b
        hypothesis = tf.nn.softmax(logits) #  [batch, output_size]
        return logits, hypothesis

def build_inputs():
    # tf Graph input
    # A placeholder for indicating each sequence length
    keep_prob = tf.placeholder(tf.float32)

    sdp_words_index = tf.placeholder(tf.int32, [None, None])
    sdp_rev_words_index = tf.placeholder(tf.int32, [None, None])
    sdp_rels_index = tf.placeholder(tf.int32, [None, None])
    sdp_rev_rels_index = tf.placeholder(tf.int32, [None, None])
    label_fb = tf.placeholder(tf.int32, [None, 19])
    label_concat = tf.placeholder(tf.int32, [None, 10])

    inputs = {
        "sdp_words_index": sdp_words_index,
        "sdp_rev_words_index": sdp_rev_words_index,
        "sdp_rels_index": sdp_rels_index,
        "sdp_rev_rels_index": sdp_rev_rels_index,
        "label_fb": label_fb,
        "label_concat": label_concat,
        }
    return inputs, keep_prob

def model(input_, word_vec_matrix_pretrained, keep_prob, config):
    #word_vec = tf.Variable(word_vec_matrix_pretrained, name="word_vec", dtype=tf.float32)
    rel_vec = tf.Variable(tf.random_uniform([config.rel_size, config.rel_vec_size], -0.05, 0.05), name="rel_vec", dtype=tf.float32)
    #rel_vec = tf.Variable(tf.random_normal([config.rel_size, config.rel_vec_size], 0.0, 0.01), name="rel_vec", dtype=tf.float32)
    word_vec = tf.constant(value=word_vec_matrix_pretrained, name="word_vec", dtype=tf.float32)
    #rel_vec = tf.constant(tf.random_uniform([config.rel_size, config.rel_vec_size], -1.0, 1.0), name="rel_vec", dtype=tf.float32)
    #tf.add_to_collection(l2_collection_name, word_vec)
    tf.add_to_collection(l2_collection_name, rel_vec)

    with tf.name_scope("look_up_table_f"):
        inputs_words_f = tf.nn.embedding_lookup(word_vec, input_["sdp_words_index"])
        inputs_rels_f = tf.nn.embedding_lookup(rel_vec, input_["sdp_rels_index"])
        inputs_words_f = tf.nn.dropout(inputs_words_f, keep_prob)
        inputs_rels_f = tf.nn.dropout(inputs_rels_f, keep_prob)

    with tf.name_scope("lstm_f"):
        words_lstm_rst_f = lstm_layer(inputs_words_f, length2(input_["sdp_words_index"]), config.word_lstm_hidden_size, config.forget_bias, "word_lstm_f")
        rels_lstm_rst_f = lstm_layer(inputs_rels_f, length2(input_["sdp_rels_index"]), config.rel_lstm_hidden_size, config.forget_bias, "rel_lstm_f")

    with tf.name_scope("conv_max_pool_f"):
        conv_output_f = conv_layer(words_lstm_rst_f, rels_lstm_rst_f, config.concat_conv_size, config.conv_out_size, "conv_f")
        pool_output_f = pool_layer(conv_output_f, config)

    with tf.name_scope("look_up_table_b"):
        inputs_words_b = tf.nn.embedding_lookup(word_vec, input_["sdp_rev_words_index"])
        inputs_rels_b = tf.nn.embedding_lookup(rel_vec, input_["sdp_rev_rels_index"])
        inputs_words_b = tf.nn.dropout(inputs_words_b, keep_prob)
        inputs_rels_b = tf.nn.dropout(inputs_rels_b, keep_prob)

    with tf.name_scope("lstm_b"):
        words_lstm_rst_b = lstm_layer(inputs_words_b, length2(input_["sdp_rev_words_index"]), config.word_lstm_hidden_size, config.forget_bias, "word_lstm_b")
        rels_lstm_rst_b = lstm_layer(inputs_rels_b, length2(input_["sdp_rev_rels_index"]), config.rel_lstm_hidden_size, config.forget_bias, "rel_lstm_b")

    with tf.name_scope("conv_max_pool_b"):
        conv_output_b = conv_layer(words_lstm_rst_b, rels_lstm_rst_b, config.concat_conv_size, config.conv_out_size, "conv_b")
        pool_output_b = pool_layer(conv_output_b, config)

    with tf.name_scope("softmax"):
        pool_concat = tf.concat([pool_output_f, pool_output_b], 1)
        logits_f, hypothesis_f = softmax_layer(pool_output_f, config.conv_out_size, 19, "softmax_f")
        logits_b, hypothesis_b = softmax_layer(pool_output_b, config.conv_out_size, 19, "softmax_b")
        logits_concat, hypothesis_concat = softmax_layer(pool_concat, 2*(config.conv_out_size), 10, "softmax_concat")

    regularizers = 0
    vars = tf.get_collection(l2_collection_name)
    for var in vars:
        regularizers += tf.nn.l2_loss(var)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_f, labels=input_["label_fb"])
    loss += tf.nn.softmax_cross_entropy_with_logits(logits=logits_b, labels=input_["label_fb"])
    loss += tf.nn.softmax_cross_entropy_with_logits(logits=logits_concat, labels=input_["label_concat"])
    loss += config.l2 * regularizers

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.grad_clip)
    #train_op = tf.train.AdamOptimizer(config.lr)
    train_op = tf.train.AdadeltaOptimizer(config.lr)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    #optimizer = tf.train.AdanOptimizer(learning_rate=config.lr).minimize(loss)

    prediction = get_prediction(hypothesis_f, hypothesis_b, config.alpha)
    correct_prediction = tf.equal(prediction, tf.argmax(input_["label_fb"], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, optimizer

def main():
    config = MyConfig()
    file_name = "data/final_data/data_" + word_vec_file_state[config.file_index] + ".pkl"
    dg = DataGenerator(file_name)
    word_vec_matrix = dg.word_vec_matrix
    # tf placeholder for input data
    inputs, keep_prob = build_inputs()
    # create bircnn model
    loss, accuracy, optimizer = model(inputs, word_vec_matrix, keep_prob, config)
    save_period = 100
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        counter = 0
        for e in range(0, config.epochs):
            while not dg.get_is_completed():
                counter += 1  # 记录经历了多少个mini_batch
                train_data = dg.next_batch(config.batch_size)
                start = time.time()
                feed = {
                    inputs["sdp_words_index"]: train_data["sdp_words_index"],
                    inputs["sdp_rev_words_index"]: train_data["sdp_rev_words_index"],
                    inputs["sdp_rels_index"]: train_data["sdp_rels_index"],
                    inputs["sdp_rev_rels_index"]: train_data["sdp_rev_rels_index"],
                    inputs["label_fb"]: train_data["label_fb"],
                    inputs["label_concat"]: train_data["label_concat"],
                    keep_prob: config.keep_prob,
                }

                train_batch_loss, train_batch_accuracy, train_ = sess.run([loss, accuracy, optimizer], feed_dict=feed)
                end = time.time()

                if counter % 20 == 0:
                    start_ = time.time()
                    valid_data = dg.get_valid_data()
                    feed = {
                        inputs["sdp_words_index"]: valid_data["sdp_words_index"],
                        inputs["sdp_rev_words_index"]: valid_data["sdp_rev_words_index"],
                        inputs["sdp_rels_index"]: valid_data["sdp_rels_index"],
                        inputs["sdp_rev_rels_index"]: valid_data["sdp_rev_rels_index"],
                        inputs["label_fb"]: valid_data["label_fb"],
                        inputs["label_concat"]: valid_data["label_concat"],
                        keep_prob: 1,
                    }
                    valid_batch_loss, valid_batch_accuracy = sess.run([loss, accuracy], feed_dict=feed)
                    end_ = time.time()

                    print("epoch: {}/{}    ".format(e+1, config.epochs),
                          "train steps: {}    ".format(counter),
                          "train accuracy: {:.4f}    ".format(train_batch_accuracy),
                          "{:.4f} sec/batch".format((end-start)))

                    print("valid accuracy: {:.4f}    ".format(valid_batch_accuracy),
                          "{:.4f} sec/valiation".format((end_-start_)))

                if counter % save_period == 0:
                    saver.save(sess, "checkpoints/counter_{}.ckpt".format(counter))

            dg.reset_is_completed()
        # finish training
        saver.save(sess, "checkpoints/counter_{}.ckpt".format(counter))


if __name__ == "__main__":
    main()

