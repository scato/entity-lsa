import logging
from pprint import pprint

from gensim import corpora, models


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# present topics
lsi = models.LsiModel.load('tmp/topic_model.lsi')
pprint(lsi.print_topics(10))
