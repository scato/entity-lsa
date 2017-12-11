import logging
from pprint import pprint

from gensim import corpora, models, matutils
import pandas as pd
from scipy.sparse import dok_matrix


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read page links from CSV
raw_page_links = pd.read_csv('tmp/frequent_page_links.csv', quotechar='\"', names=('from', 'to'), na_filter=False)
page_links = raw_page_links.as_matrix()

# create dictionary
dictionary = corpora.Dictionary.load('tmp/page_links.dict')

# create corpus
corpus_keys = [
    (
        dictionary.token2id[page_link[1]],
        dictionary.token2id[page_link[0]]
    )
    for page_link in page_links
]
corpus_size = max(dictionary.iterkeys()) + 1

corpus_dok = dok_matrix((corpus_size, corpus_size))
for corpus_key in corpus_keys:
    corpus_dok[corpus_key] = 1

corpus_csc = corpus_dok.tocsc()
corpus = matutils.Sparse2Corpus(corpus_csc)

# control for DF
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

corpora.MmCorpus.serialize('tmp/page_links.mm', corpus_tfidf)
