import logging

from gensim import corpora, models, similarities


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load corpus, dictionary and model
corpus = corpora.MmCorpus('tmp/page_links.mm')
lsi = models.LsiModel.load('tmp/topic_model.lsi')

# create similarity matrix
index = similarities.MatrixSimilarity(lsi[corpus])
index.save('tmp/page_links.index')
