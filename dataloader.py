import numpy as np
from word2index import Word2index
from keras import preprocessing


def str2idxs(sents, w2x):
    for line_ind, line in enumerate(sents):
        for words_ind, words in enumerate(line):
                sents[line_ind][words_ind]= w2x(words)

    return sents

def padding_data(seqs, w2x):
        seqs = preprocessing.sequence.pad_sequences(seqs, dtype='int32', padding='pre', truncating='pre',value=w2x.PAD_IDX)
        return seqs


class Gen_Data_loader():
    def __init__(self, word_dict,  batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.word_dict = word_dict#Word2index()
        #self.word_dict.load_dict('save/word2indx.txt')

    def create_batches(self, data_file, gen_flag=0):
        self.token_stream = []
        first = True
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [str(x) for x in line]

                if first:  # remove \ufeff special character from the first line
                    parse_line[len(parse_line)-1] = parse_line[len(parse_line)-1].replace('\ufeff', '')
                    first = False

                if 3 <= len(parse_line) <= 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]

        # convert words to indices and then pad them to have the same length
        if gen_flag:
            self.token_stream = str2idxs(self.token_stream, self.word_dict)
            self.token_stream = padding_data(self.token_stream, self.word_dict)

        self.pointer = 0

    def next_batch(self):

        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch

        ret = np.array([np.array(xi) for xi in ret])

        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, word_dict, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.word_dict = word_dict #Word2index()
        #self.word_dict.load_dict('save/word2indx.txt')

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file, encoding='utf-8')as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [str(x) for x in line]
                if 3 <= len(parse_line) <= 20:
                    positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if 3 <= len(parse_line) <= 20:
                    negative_examples.append(parse_line)

        positive_examples = str2idxs(positive_examples, self.word_dict)
        positive_examples = padding_data(positive_examples, self.word_dict)
        self.sentences = np.concatenate([positive_examples, negative_examples], 0)

        # print(len(positive_examples), len(positive_examples[0]))
        # print(len(negative_examples), len(negative_examples[0]))

        #self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
