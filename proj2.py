import numpy as np
import scipy.optimize
from sklearn import svm
from sklearn import linear_model
import gzip
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
import string; import array; import random
import matplotlib.pyplot as plt
import time; import calendar
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


def parseData(fname):
  for l in open(fname):
    yield eval(l)

def readJSON(path):
  for l in open(path, 'rt'):
    d = eval(l)
    u = d['userID']
    try:
      g = d['gameID']
    except Exception as e:
      g = None
    yield u,g,d

null=None; true=True; false=False

data_raw = list(parseData('renttherunway_final_data.json'))
data = []
for d in data_raw:
    d = {x.replace(' ', '_'): v  for x, v in d.items()} 
    data.append(d)
#%% Task1: Basic Statistics
purposes = [] 
for d in data:
    if 'rented_for' in d:
        purposes.append(d['rented_for'])
    else:
        purposes.append('wedding')
    
counter = Counter(purposes)

labels, values = zip(*counter.items())
width = 0.2
plt.bar(labels, values, width)
plt.show()

#%%
purposeID = dict(zip([d[0] for d in counter.items()], range(len(counter))))
monthDict = dict((month, index) for index, month in enumerate(calendar.month_abbr) if month)
dates_Wedding = [d['review_date'][:3] for d in data if 'rented_for' in d and d['rented_for']=='wedding']
counter = Counter(dates_Wedding)
stat = dict()        # Month -- temporal feature?
for c in counter:
    stat[monthDict[c]]=counter[c]

x, y = zip(*sorted(stat.items())) # unpack a list of pairs into two tuples
plt.bar(x, y)
plt.show()

# Data clearning (preprocessing): Stemming and removing extremes
def lemmatize_stemming(text):
    stemmer = nltk.stem.porter.PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = [preprocess(d) for d in documents]
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Top 200 most popular words according to document frequency
wordCount_dict = defaultdict(int)
for d in processed_docs:
    for w in d:
        wordCount_dict[w] += 1
counts = [(wordCount_dict[w], w) for w in wordCount_dict]
counts.sort(reverse = True)
# pprint(counts[:200])

#%%
X = 



#%% Attempt LDA Model
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
#%% LDA - BOW
topic_dic = defaultdict(int)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    topic_dic[topic] = idx
     
for i in range(50):
    for index, score in sorted(lda_model[bow_corpus[i]], key=lambda tup: -1*tup[1]):
        top = topic_dic[lda_model.print_topic(index, 10)]
        print(top)
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
        break;

results = []
for i in range(len(processed_docs)):
    for index, score in sorted(lda_model[bow_corpus[i]], key=lambda tup: -1*tup[1]):
        top = topic_dic[lda_model.print_topic(index, 10)]
        results.append(top)    
        break;
        
for i in range(100):
    print("Score: {}\t Topic: {}".format(results[i], purposes[i]))
    
#%% LDA - TFIDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=2, workers=4)

topic_dic = defaultdict(int)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    topic_dic[topic] = idx
     
results = []
for i in range(len(processed_docs)):
    for index, score in sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1]):
        top = topic_dic[lda_model.print_topic(index, 10)]
        results.append(top)    
        break;
        
for i in range(100):
    print("Score: {}\t Topic: {}".format(results[i], purposes[i]))

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        