import logging
from pprint import pprint

from gensim import corpora, models, similarities


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load corpus, dictionary, model and index
corpus = corpora.MmCorpus('tmp/page_links.mm')
dictionary = corpora.Dictionary.load('tmp/page_links.dict')
lsi = models.LsiModel.load('tmp/topic_model.lsi')
index = similarities.MatrixSimilarity.load('tmp/page_links.index')

# present topics
pprint(lsi.print_topics(5))

# present the first entity
entity = dictionary[0]
vec = corpus[0]
doc = [(dictionary[id], weight) for id, weight in vec]
pprint((entity, doc))

# use similarity matrix
vec_lsi = lsi[vec]
sims = index[vec_lsi]
ordered_sims = sorted(enumerate(sims), key=lambda item: -item[1])[:10]
results = [(dictionary[sim[0]], sim[1]) for sim in ordered_sims]
pprint(results)
