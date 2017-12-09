from collections import defaultdict
import logging
import csv

import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read page links from CSV
raw_page_links = pd.read_csv('page_links_nl.csv', quotechar='\"', names=('from', 'to'), na_filter=False)
page_links = raw_page_links.as_matrix()

# remove pages with a DF less than 5
frequency = defaultdict(int)
for page_link in page_links:
    frequency[page_link[1]] += 1

page_links = [
    page_link
    for page_link in page_links
    if frequency[page_link[1]] >= 5
]

# write page links to CSV
pd.DataFrame(page_links).to_csv('tmp/frequent_page_links.csv', quoting=csv.QUOTE_ALL, quotechar='\"', header=False, index=False)
