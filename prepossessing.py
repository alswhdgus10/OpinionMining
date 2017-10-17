def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # header 제외
    return data


from konlpy.tag import Twitter

pos_tagger = Twitter()

meaningless_tag = ["Josa", "Punctuation", "Determiner"]
common_word = ["하다", "되다", "보다", "없다", "아니다", "이다"]


def tokenize(doc):  # norm, stem은 optional
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True)
            if t[1] not in meaningless_tag and t[0] not in common_word ]


def get_bi_gram(str):
    tokenzied = tokenize(str);
    return zip(tokenzied, tokenzied[1:])

train_data = read_data('rsrc/ratings_train.txt')[0:1000]
test_data = read_data('rsrc/ratings_test.txt')[0:100]
train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
#train_docs_bi_gram = [(get_bi_gram(row[1]), row[2]) for row in train_data]
#test_docs_bi_gram = [(get_bi_gram(row[1]), row[2]) for row in test_data]



import nltk

tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(200)]


def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}


# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음


train_docs = train_docs
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)


selected_words

print(nltk.classify.accuracy(classifier, test_xy))
