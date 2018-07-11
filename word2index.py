from collections import defaultdict


UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


class Word2index(object):

    def __init__(self):
        self.dict = {}


    def create_dict(self, input_file, output_file):
        self.fout = open(output_file, 'w', encoding='utf-8')
        self.fin = open(input_file, 'r', encoding='utf-8')

        # writing first the reserved words in the text file
        self.fout.write('%s\n' % SOS_TOKEN)  # start token
        self.fout.write('%s\n' % EOS_TOKEN)  # start token
        self.fout.write('%s\n' % PAD_TOKEN)  # start token
        self.fout.write('%s\n' % UNK_TOKEN)  # start token


        vocab = set()

        for line in self.fin:
            words = line.rsplit()
            for word in words:
                if not word in vocab:
                    self.fout.write(word + '\n')
                    vocab.add(word)

    def load_dict(self, word_file):
        self.fin = open(word_file, 'r', encoding='utf-8')
        self.count = 0

        self.dict = {}
        first = True
        for word in self.fin:
            word = word[0:-1]   # remove \n

            if(first):
                word = word.replace('\ufeff', '')
                first = False

            self.dict[word] = self.count
            self.count += 1

        self.UNK_IDX = self.dict[UNK_TOKEN]
        self.PAD_IDX = self.dict[PAD_TOKEN]
        self.SOS_IDX = self.dict[SOS_TOKEN]
        self.EOS_IDX = self.dict[EOS_TOKEN]

    def __len__(self):
        return len(self.dict)

    def __call__(self, word):
        return self.dict.get(word, self.UNK_IDX)
