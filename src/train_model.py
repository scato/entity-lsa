import logging

from gensim import corpora, models


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load corpus and dictionary
corpus = corpora.MmCorpus('tmp/page_links.mm')[:20000]
dictionary = corpora.Dictionary.load('tmp/page_links.dict')

# train model
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=50)
lsi.save('tmp/topic_model.lsi')
