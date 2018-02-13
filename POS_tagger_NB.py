from __future__ import division
import os
from nltk import tuple2str
from sklearn.metrics import accuracy_score
from nltk.corpus import brown

brown_tagged_sent = brown.tagged_sents()
def transfrom_data(data):
    sentences= []
    word = ''
    for sent in data:
        for tuple_word_tag in sent:
            str = tuple2str(tuple_word_tag)
            word += str
            word += ' '
        sentences.append(word)
        word = ' '
    return sentences

brown_sent = transfrom_data(brown_tagged_sent)
total_word = ''
for sent in brown_sent:
    for word_tag in sent.split():
        total_word += word_tag
        total_word += ' '
total_word = total_word.split()
V = len(set(total_word))

size = int(len(brown_tagged_sent)*0.7)
train_brown_sent = transfrom_data(brown_tagged_sent[:size])
test_brown_sent = transfrom_data(brown_tagged_sent[size:])

test_sentences = []
test_tags = ''
test_word = ''
test_tag = ''
for sent in test_brown_sent:
    for word_tag in sent.split():
        word_tag_broken = word_tag.split('/')
        wordBroken = word_tag_broken[0]
        tagBroken = word_tag_broken[1]
        test_word += wordBroken
        test_word += ' '
        test_tag += tagBroken
        test_tag += ' '
    test_sentences.append(test_word)
    test_tags += test_tag
    test_word = ''
    test_tag = ''
test_tags = test_tags.split()

def filter_text(text):
    for word in text.split():
        if ('/' not in word):
            text = text.replace(word, '')
    return text

num_of_tag = {}  # Dict lưu trữ số lần xuất hiện của tag trong ngữ liệu. VD: {'NN': 3000, 'VB': 2500, ...}
num_of_word_tag = {}  # Dict lưu trữ số lần xuất hiện của các cặp word/tag. VD: {'She/PPS': 120, 'looked/VBD': 40, ...}

for sent in train_brown_sent:
    words = sent.split()
    for word_tag in words:
        word_tag = word_tag.lower()
        if word_tag not in num_of_word_tag.keys():
            num_of_word_tag[word_tag] = 1
        else:
            num_of_word_tag[word_tag] += 1
        word_tag_broken = word_tag.split('/')
        word = word_tag_broken[0]
        tag = word_tag_broken[1]

        if tag not in num_of_tag.keys():
            num_of_tag[tag] = 1
        else:
            num_of_tag[tag] += 1

print('OK')

sum_num_of_tag = sum(num_of_tag.values())
tag_predict = ''

for sent in test_sentences:
    words = sent.split()
    for word in words:
        word = word.lower()
        max_prob_of_tag = 0
        predict_tag = []
        for tag in num_of_tag.keys():
            prior = num_of_tag[tag] / sum_num_of_tag
            word_tag = word + '/' + tag
            if word_tag in num_of_word_tag.keys():
                likelihood = (num_of_word_tag[word_tag] + 1) / (num_of_tag[tag] + V)
            else:
                likelihood = 1 / (num_of_tag[tag] + V)
            prob_tag_word = prior * likelihood

            if prob_tag_word > max_prob_of_tag:
                predict_tag = []
                predict_tag.append(tag)
                max_prob_of_tag = prob_tag_word
        tag_predict += predict_tag[0].upper()
        tag_predict += ' '
tag_predict = tag_predict.split()

print(accuracy_score(test_tags, tag_predict))