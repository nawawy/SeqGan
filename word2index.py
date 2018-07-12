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
        self.dict_indx = {}

        first = True
        for word in self.fin:
            word = word[0:-1]   # remove \n

            if(first):
                word = word.replace('\ufeff', '')
                first = False

            self.dict[word] = self.count
            self.dict_indx[self.count] = word

            self.count += 1

        self.UNK_IDX = self.dict[UNK_TOKEN]
        self.PAD_IDX = self.dict[PAD_TOKEN]
        self.SOS_IDX = self.dict[SOS_TOKEN]
        self.EOS_IDX = self.dict[EOS_TOKEN]

        self.dict_indx[self.UNK_IDX] = UNK_TOKEN
        self.dict_indx[self.PAD_IDX] = PAD_TOKEN
        self.dict_indx[self.SOS_IDX] = SOS_TOKEN
        self.dict_indx[self.EOS_IDX] = EOS_TOKEN

    def indx_to_word(self, indx):
        return self.dict_indx.get(indx)

    def __len__(self):
        return len(self.dict)

    def __call__(self, word):
        return self.dict.get(word, self.UNK_IDX)

w = Word2index()
w.load_dict('save/word2indx.txt')
f = open('save/generator_sample.txt', 'r')
fout = open('output_arabic.txt', 'w', encoding=
            'utf-8')
for line in f:
    line = line.strip()
    line = line.split()
    parse_line = [str(w.indx_to_word(int(x))) for x in line]
    out = ''
    for word in parse_line:
        out += word + ' '
    fout.write('%s\n' % out)

