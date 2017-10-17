from functools import reduce

import nltk

statements = {
    ("연출을 못하네", 0),
    ("연기를 못한다", 0),
    ("시간 아깝다", 0),
    ("돈이 아깝다", 0),
    ("시간 버렸다", 0),
    ("연기 정말 못한다", 0),
    ("정말 후회한다", 0),
    ("난 재미없던데", 0),
    ("진짜 별로네", 0),
    ("핵노잼이야", 0),
    ("진심 짜증났다", 0),
    ("정말 재미없다", 0),
    ("원작보다 못하다", 0),
    ("못 쓸 영화네", 0),
    ("재밌는 영화다", 1),
    ("정말 재밌었어요", 1),
    ("매력적인 영화다", 1),
    ("시간 가는줄 몰랐다", 1),
    ("정말 최고의 영화다", 1),
    ("완전 추천한다", 1),
    ("긴장감이 넘친다", 1),
    ("훌륭한 영화다", 1),
    ("정말 재미있다", 1),
    ("배우가 매력적이다", 1),
    ("원작보다 좋다", 1),
    ("연출 괜찮다", 1),
    ("라인업이 화려하네", 1),
    ("인정하는 부분", 1),
    ("연기를 정말 잘한다", 1)}

from konlpy.tag import Twitter

pos_tagger = Twitter()

meaningless_tag = ["Josa", "Punctuation", "Determiner", "Suffix"]
common_word = ["하다", "되다", "보다", "없다", "아니다", "이다"]


def tokenize(doc):  # norm, stem은 optional
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True)
            if t[1] not in meaningless_tag and t[0] not in common_word]


train_docs = [(tokenize(row[0]), row[1]) for row in statements]
tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(50)]


def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}


# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음

train_docs = train_docs
train_xy = [(term_exists(d), c) for d, c in train_docs]

total_num = len(train_xy)
positive = ([x for x in train_xy if x[1] == 1])
negative = ([x for x in train_xy if x[1] == 0])

p_positive = len(positive) / total_num
p_negative = len(negative) / total_num

positive_word_count = []
negative_word_count = []

positive_total = 0
negative_total = 0

for w in selected_words:
    doc_including_w_positive = [doc for doc in positive if doc[0]['exists({})'.format(w)]]
    positive_cnt = len(doc_including_w_positive) + 0.01
    positive_word_count.append((w, positive_cnt))
    positive_total += positive_cnt

    doc_including_w_negative = [doc for doc in negative if doc[0]['exists({})'.format(w)]]
    negative_cnt = len(doc_including_w_negative) + 0.01
    negative_word_count.append((w, negative_cnt))
    negative_total += negative_cnt

positive_word_prob = [(cnt[0], cnt[1] / positive_total) for cnt in positive_word_count]
negative_word_prob = [(cnt[0], cnt[1] / negative_total) for cnt in negative_word_count]


def guess(doc):
    preprocessed = tokenize(doc)
    feature = term_exists(preprocessed)

    matched_positive = [p for (word, p) in positive_word_count if feature['exists({})'.format(word)]]
    if len(matched_positive) != 0:
        p_doc_over_positive = reduce(lambda a, b: a * b,matched_positive)
    else:
        p_doc_over_positive = 0

    matched_negative = [p for (word, p) in negative_word_count if feature['exists({})'.format(word)]]
    if len(matched_negative) != 0:
        p_doc_over_negative = reduce(lambda a, b: a * b, matched_negative)
    else:
        p_doc_over_negative = 0

    p_positive_over_doc = p_doc_over_positive * p_positive
    p_negative_over_doc = p_doc_over_negative * p_negative

    if p_positive_over_doc>p_negative_over_doc:
        return "positive"
    elif p_positive_over_doc<p_negative_over_doc:
        return "negative"
    else:
        return "neutral"


while True:
    text = input()
    label = guess(text)
    print(label)
