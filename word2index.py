UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


class Word2index(object):

    def __init__(self):
        self.dict = {}
        self.count = 0

    def create_dict(self, input_file, output_file):
        """
        :param input_file: data input file path
        :param output_file: vocab wors listed in a file for reference
        """
        self.fout = open(output_file, 'w', encoding='ISO-8859-1')
        self.fin = open(input_file, 'r', encoding='ISO-8859-1')  #ISO-8859-1

        # writing first the reserved words in the text file
        self.fout.write('%s\n' % SOS_TOKEN)  # start token
        self.fout.write('%s\n' % EOS_TOKEN)  # start token
        self.fout.write('%s\n' % PAD_TOKEN)  # start token
        self.fout.write('%s\n' % UNK_TOKEN)  # start token

        self.dict[SOS_TOKEN] = 0
        self.dict[EOS_TOKEN] = 1
        self.dict[PAD_TOKEN] = 2
        self.dict[UNK_TOKEN] = 3

        vocab = set()

        self.count = 4

        for line in self.fin:
            words = line.rsplit()
            for word in words:
                if word not in vocab:
                    self.fout.write(word + '\n')
                    self.dict[word] = list([self.count, 1])
                    self.count += 1
                    vocab.add(word)
                else:
                    self.dict[word][1] += 1

    def remove_least_freq(self, input_file, threshold):
        """
        :param input_file: name of data input file
        :param threshold: threshold of the frequency of the word to remove
        """
        fin = open(input_file, 'r+', encoding='ISO-8859-1')
        lines = fin.readlines()
        for line_indx, line in enumerate(lines):
            print('Line : ', line_indx)
            words = line.rsplit()
            for word in words:
                if self.dict[word][1] <= threshold:
                    del lines[line_indx]
                    break
        fout = open('ar_filtered', 'w', encoding='ISO-8859-1')
        fout.truncate(0)
        fout.writelines(lines)

    def load_dict(self, word_file):
        """

        :param word_file: vocab file path
        :return: load the self.dict member dictionary
        """
        self.fin = open(word_file, 'r', encoding='ISO-8859-1')
        self.count = 0

        self.dict = {}
        self.dict_indx = {}

        first = True
        for word in self.fin:
            word = word[0:-1]  # remove \n

            if (first):
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
