import logging

from gensim import corpora
import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read page links from CSV
raw_page_links = pd.read_csv('tmp/frequent_page_links.csv', quotechar='\"', names=('from', 'to'), na_filter=False)
page_links = raw_page_links.as_matrix()

# create dictionary
dictionary = corpora.Dictionary(page_links, prune_at=None)
dictionary.save('tmp/page_links.dict')
