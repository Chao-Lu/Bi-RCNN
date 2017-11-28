class MyConfig(object):
    file_index = 3
    lr = 0.1
    alpha = 0.5
    l2 = 10e-4
    keep_prob = 0.5
    lstm_out_keep_prob = 1
    batch_size = 128
    grad_clip = 1000
    epochs = 2000

    # may change
    forget_bias = 1.0
    word_vec_size = 300
    rel_vec_size = 30

    conv_out_size = 200
    word_lstm_hidden_size = word_vec_size
    rel_lstm_hidden_size = rel_vec_size
    concat_conv_size = 2*word_lstm_hidden_size + rel_lstm_hidden_size

    # never change
    n_classes = 10
    rel_size = 29


