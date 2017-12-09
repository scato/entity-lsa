import logging
from pprint import pprint

from gensim import corpora, models, matutils
import pandas as pd
from scipy.sparse import dok_matrix


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load corpus and dictionary
corpus = corpora.MmCorpus('tmp/page_links.mm')[:1000]
dictionary = corpora.Dictionary.load('tmp/page_links.dict')

# control for DF
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# train model
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
pprint(lsi.print_topics(10))