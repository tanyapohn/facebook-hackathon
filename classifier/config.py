class Config(object):
    N = 1  # 6 in Transformer Paper
    d_model = 256  # 512 in Transformer Paper
    d_ff = 512  # 2048 in Transformer Paper
    h = 4
    dropout = 0.1
    output_size = 2
    lr = 0.3
    max_epochs = 35
    batch_size = 32
    max_sen_len = 600
