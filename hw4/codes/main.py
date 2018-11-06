import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
import sys
import json
import time
import random
random.seed(1229)

from model import RNN, _START_VOCAB

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 18430, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 5, "Number of labels.")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 20, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_string("model", "lstm", "Choose model: [rnn, lstm, gru]")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    print('Creating %s dataset...' % fname)
    data = []
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            tokens = line.split(' ')
            data.append({'label':tokens[0], 'text':tokens[1:]})
    return data

def build_vocab(path, data):
    print("Creating vocabulary...")
    vocab = {}
    for i, pair in enumerate(data):
        for token in pair['text']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]
    else:
        FLAGS.symbols = len(vocab_list)

    print("Loading word vectors...")
    vectors = {}
    with open('%s/vector.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = list(map(float, vectors[word].split()))
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    return vocab_list, embed

def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l-len(sent))

    max_len = max([len(item['text']) for item in data])
    texts, texts_length, labels = [], [], []
        
    for item in data:
        texts.append(padding(item['text'], max_len))
        texts_length.append(len(item['text']))
        labels.append(int(item['label']))

    batched_data = {'texts': np.array(texts), 'texts_length':texts_length, 'labels':labels}

    return batched_data

def train(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = model.train_step(sess, batch_data)
        loss += outputs[0]
        accuracy += outputs[1]

    return loss / len(dataset), accuracy / len(dataset)

def evaluate(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = sess.run(['loss:0', 'accuracy:0'], {'texts:0':batch_data['texts'], 'texts_length:0':batch_data['texts_length'], 'labels:0':batch_data['labels']})
        loss += outputs[0]
        accuracy += outputs[1]
    return loss / len(dataset), accuracy / len(dataset)

def inference(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    result = []
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs = sess.run(['predict_labels:0'], {'texts:0':batch_data['texts'], 'texts_length:0':batch_data['texts_length']})
        result += outputs[0].tolist()

    with open('result.txt', 'w') as f:
        for label in result:
            f.write('%d\n' % label)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        print(FLAGS.__flags)
        data_train = load_data(FLAGS.data_dir, 'train.txt')
        data_dev = load_data(FLAGS.data_dir, 'dev.txt')
        vocab, embed = build_vocab(FLAGS.data_dir, data_train)
        
        model = RNN(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                FLAGS.labels,
                embed,
                learning_rate=FLAGS.learning_rate,
                model_choose=FLAGS.model)
        if FLAGS.log_parameters:
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)

        for epoch in list(range(FLAGS.epoch)):
            random.shuffle(data_train)
            start_time = time.time()
            loss, accuracy = train(model, sess, data_train)
            print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, model.learning_rate.eval(), time.time()-start_time, loss, accuracy))
            model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=epoch)
            loss, accuracy = evaluate(model, sess, data_dev)
            print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
    else:
        data_dev = load_data(FLAGS.data_dir, 'dev.txt')
        data_test = load_data(FLAGS.data_dir, 'test.txt')

        model = RNN(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                FLAGS.labels,
                embed=None,
                model_choose=FLAGS.model)

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)

        loss, accuracy = evaluate(model, sess, data_dev)
        print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))

        inference(model, sess, data_test)
        print("        test_set, write inference results to result.txt")


