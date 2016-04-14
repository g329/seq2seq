#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import numpy as np
import sys
import codecs

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

def make_vocab_dict(vocab):
    id2word = {}
    word2id = {}
    for id, word in enumerate(vocab):
        id2word[id] = word
        word2id[word] = id
    return id2word, word2id


class Seq2Seq(chainer.Chain):
    dropout_ratio = 0.5

    def __init__(self, input_vocab, output_vocab, feature_num, hidden_num):
        """
        :param input_vocab: array of input  vocab
        :param output_vocab: array of output  vocab
        :param feature_num: size of feature layer
        :param hidden_num: size of hidden layer
        :return:
        """
        self.id2word_input, self.word2id_input = make_vocab_dict(input_vocab)
        self.id2word_output, self.word2id_output = make_vocab_dict(output_vocab)
        self.input_vocab_size = len(self.word2id_input)
        self.output_vocab_size = len(self.word2id_output)

        super(Seq2Seq, self).__init__(
                # encoder
                word_vec=L.EmbedID(self.input_vocab_size, feature_num),
                input_vec=L.LSTM(feature_num, hidden_num),

                # connect layer
                context_lstm=L.LSTM(hidden_num, self.output_vocab_size),

                # decoder
                output_lstm=L.LSTM(self.output_vocab_size, self.output_vocab_size),
                out_word=L.Linear(self.output_vocab_size, self.output_vocab_size),
        )

    def encode(self, src_text, train):
        """

        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector
        """
        for word in src_text:
            word = chainer.Variable(np.array([[word]], dtype=np.int32))
            embed_vector = F.tanh(self.word_vec(word))
            input_feature = self.input_vec(embed_vector)
            context = self.context_lstm(F.dropout(input_feature, ratio=self.dropout_ratio, train=train))

        return context

    def decode(self, context, teacher_embed_id, train):
        """
        :param context: context vector which made `encode` function
        :param teacher_embed_id : embed id ( teacher's )
        :return: decoded embed vector
        """

        output_feature = self.output_lstm(context)
        predict_embed_id = self.out_word(output_feature)
        if train:
            t = np.array([teacher_embed_id], dtype=np.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_embed_id, t), predict_embed_id
        else:
            return predict_embed_id

    def initialize(self):
        """
        state initialize

        :param image_feature:
        :param train:
        :return:
        """
        self.input_vec.reset_state()
        self.context_lstm.reset_state()
        self.output_lstm.reset_state()

    def generate(self, start_word_id, sentence_limit):

        context = self.encode([start_word_id], train=False)
        sentence = ""

        for _ in range(sentence_limit):
            context = self.decode(context, teacher_embed_id=None, train=False)
            word = self.id2word_output[np.argmax(context.data)]
            if word == "<eos>":
                break
            sentence = sentence + word + " "
        return sentence


if __name__ == "__main__":

    input_vocab = ["<start>", u"黄昏に", u"天使の声", u"響く時，", u"聖なる泉の前にて", u"待つ", "<eos>"]
    output_vocab = [u"5時に", u"噴水の前で", u"待ってます", "<eos>"]

    model = Seq2Seq(input_vocab, output_vocab, feature_num=4, hidden_num=10)

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    for _ in range(20000):

        model.initialize()
        # reverse すると収束が早くなる
        input = [model.word2id_input[word] for word in reversed(input_vocab)]

        context = model.encode(input, train=True)
        acc_loss = 0

        for word in output_vocab:
            id = model.word2id_output[word]
            loss, context = model.decode(context, id, train=True)
            acc_loss += loss

        model.zerograds()
        acc_loss.backward()
        acc_loss.unchain_backward()
        optimizer.update()
        start = model.word2id_input["<start>"]
        sentence = model.generate(start, 7)

        print "teacher : ", "".join(input_vocab[1:6])
        print "-> ", sentence
        print
