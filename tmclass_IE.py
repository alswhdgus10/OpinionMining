import nltk
import numpy

# locs = [('Omnicom', 'IN', 'New York'),
#         ('DDB Needham', 'IN', 'New York'),
#         ('Kaplan Thaler Group', 'IN', 'New York'),
#         ('BBDO South', 'IN', 'Atlanta'),
#         ('Georgia-Pacific', 'IN', 'Atlanta')]
# query = [e1 for (e1, rel, e2) in locs if e2=='Atlanta']
# print(query)

# def ie_preprocess(document):
#     sentences = nltk.sent_tokenize(document) [1]
#     sentences = [nltk.word_tokenize(sent) for sent in sentences] [2]
#     sentences = [nltk.pos_tag(sent) for sent in sentences] [3]
#
# sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
# grammar = "NP: {<DT>?<JJ>*<NN>}"
# cp = nltk.RegexpParser(grammar)
# result = cp.parse(sentence)
# print(result)


#
# nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
# grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
# cp = nltk.RegexpParser(grammar)
# print(cp.parse(nouns))

# cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
# brown = nltk.corpus.brown
# for sent in brown.tagged_sents():
#     tree = cp.parse(sent)
#     for subtree in tree.subtrees():
#         if subtree.label() == 'CHUNK': print(subtree)

# grammar = r"""
#   NP:
#     {<.*>+}          # Chunk everything
#     }<VBD|IN>+{      # Chink sequences of VBD and IN
#   """
# sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
#        ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
# cp = nltk.RegexpParser(grammar)
# print(cp.parse(sentence))


# from nltk.corpus import conll2000


# import os
#
# megam_path = os.path.expanduser("~/megam-64.opt")
# nltk.config_megam(megam_path)
# cp = nltk.RegexpParser("")
# test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# print(cp.evaluate(test_sents))

# grammar = r"NP: {<[CDJNP].*>+}"
# cp = nltk.RegexpParser(grammar)
# print(cp.evaluate(test_sents))

# class UnigramChunker(nltk.ChunkParserI):
#     def __init__(self, train_sents):
#         train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
#                       for sent in train_sents]
#         self.tagger = nltk.UnigramTagger(train_data)
#
#     def parse(self, sentence):
#         pos_tags = [pos for (word, pos) in sentence]
#         tagged_pos_tags = self.tagger.tag(pos_tags)
#         chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
#         conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
#                      in zip(sentence, chunktags)]
#         return nltk.chunk.conlltags2tree(conlltags)
#
#
# test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
# train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
# unigram_chunker = UnigramChunker(train_sents)
# # print(unigram_chunker.evaluate(test_sents))
#
# postags = sorted(set(pos for sent in train_sents
#                      for (word, pos) in sent.leaves()))


# print(unigram_chunker.tagger.tag(postags))

# class ConsecutiveNPChunkTagger(nltk.TaggerI):
#     def __init__(self, train_sents):
#         train_set = []
#         for tagged_sent in train_sents:
#             untagged_sent = nltk.tag.untag(tagged_sent)
#             history = []
#             for i, (word, tag) in enumerate(tagged_sent):
#                 featureset = npchunk_features(untagged_sent, i, history)
#                 train_set.append((featureset, tag))
#                 history.append(tag)
#         self.classifier = nltk.MaxentClassifier.train(
#             train_set, algorithm='IIS', trace=0)
#
#     def tag(self, sentence):
#         history = []
#         for i, word in enumerate(sentence):
#             featureset = npchunk_features(sentence, i, history)
#             tag = self.classifier.classify(featureset)
#             history.append(tag)
#         return zip(sentence, history)
#
#
# class ConsecutiveNPChunker(nltk.ChunkParserI):
#     def __init__(self, train_sents):
#         tagged_sents = [[((w, t), c) for (w, t, c) in
#                          nltk.chunk.tree2conlltags(sent)]
#                         for sent in train_sents]
#         self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
#
#     def parse(self, sentence):
#         tagged_sents = self.tagger.tag(sentence)
#         conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
#         return nltk.chunk.conlltags2tree(conlltags)
#

# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     return {"pos": pos}
#
#
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))

# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i-1]
#     return {"pos": pos, "prevpos": prevpos}
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))

# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i-1]
#     return {"pos": pos, "word": word, "prevpos": prevpos}
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))
#
# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i - 1]
#     if i == len(sentence) - 1:
#         nextword, nextpos = "<END>", "<END>"
#     else:
#         nextword, nextpos = sentence[i + 1]
#     return {"pos": pos,
#             "word": word,
#             "prevpos": prevpos,
#             "nextpos": nextpos,
#             "prevpos+pos": "%s+%s" % (prevpos, pos),
#             "pos+nextpos": "%s+%s" % (pos, nextpos),
#             "tags-since-dt": tags_since_dt(sentence, i)}
#
#
# def tags_since_dt(sentence, i):
#     tags = set()
#     for word, pos in sentence[:i]:
#         if pos == 'DT':
#             tags = set()
#         else:
#             tags.add(pos)
#     return '+'.join(sorted(tags))
#
#
# chunker = ConsecutiveNPChunker(train_sents)
#
# print(chunker.evaluate(test_sents))

# nltk.download()
# sent = nltk.corpus.treebank.tagged_sents()[22]
# print(nltk.ne_chunk(sent, binary=True))

import re

from nltk.corpus import conll2002
vnv = """
(
is/V|    # 3rd sing present and
was/V|   # past forms of the verb zijn ('be')
werd/V|  # and also present
wordt/V  # past of worden ('become)
)
.*       # followed by anything
van/Prep # followed by van ('of')
"""
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for r in nltk.sem.extract_rels('PER', 'ORG', doc,
                                   corpus='conll2002', pattern=VAN):
        print(nltk.sem.clause(r, relsym="VAN"))
