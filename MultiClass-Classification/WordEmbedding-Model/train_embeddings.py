# Train a word2vec model and save it for later use

import gensim
import os
import re
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize


# For more info on gensim and its usage: http://rare-technologies.com/deep-learning-with-word2vec-and-gensim/

class BlogSentences(object):

    def __init__(self, directory):
        # To keep a count
        self.i = 0

        # directory is the path to the .txt files
        self.directory = directory

    # Iterator for the class
    # Returns a list of words (a sentence)
    def __iter__(self):
        for D in os.listdir(self.directory)[1:]:
            for txtFile in os.listdir(self.directory+'/'+D):
                for line in open(self.directory+'/'+D+'/'+txtFile, 'rb'):
                    for sentence in line_to_sents(line):
                        words = sent_to_words(sentence)

                        # Ignore really short sentences
                        if len(words) > 3:
                            yield words

                if self.i % 1000 == 0:
                    print("Files: ", self.i)
                self.i += 1


# Converts line to sentences
def line_to_sents(line):
    sentences = sent_tokenize(line.decode('utf8'))
    for s in sentences:
        # Remove weird characters
        s = re.sub(r'[^a-zA-Z]', " ", s)
        yield s

# Converts sentences to words
def sent_to_words(sentence):
    # Make lowercase
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    my_stopwords = stopwords.words('english') + list(punctuation)
    # Filter stopwords
    words_output = []
    for w in words:
        if w not in my_stopwords:
            words_output.append(w)

    return words_output

print("Total 154924 files. Should run 6x times.")
path = '/Users/dh_lab_05/Desktop/txtLAB/Dataset'
s = BlogSentences(path)
model = gensim.models.Word2Vec(s, size=300, workers=8, min_count=10)
model.save('/Users/dh_lab_05/Desktop/ECCO_model.word2vec')
