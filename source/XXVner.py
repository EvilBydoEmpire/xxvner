#!/usr/bin/env python
# coding: utf-8

# ## XXVner

# Multilingual Named Entity Recognition with One Model

# loading the .ipynb file in a Jupyter Notebook Environment is recommended!


# #### Setup

# import libraries
get_ipython().system('pip install flair')
import flair

import flair.datasets
from flair.data import Corpus
from flair.data import MultiCorpus
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# specify storage
folder = '' # change to '/your_folder/XXVner/corpora'

# specify setting for XXVcorpus
corpora = []
downsample = False
inplace = False


# #### Corpora

corpus = flair.datasets.EUROPARL_NER_GERMAN(in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.7)
europarl = corpus
corpora.append(europarl)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[10].to_tagged_string('ner'))


# In[ ]:


corpus = flair.datasets.WIKIGOLD_NER(in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.9)
wikigold = corpus
corpora.append(wikigold)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'dane'
columns = {0: 'id', 1:'text', 2:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.9)
dane = corpus
corpora.append(dane)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'basque'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.7)
basque = corpus
corpora.append(basque)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'bokmal'
columns = {0: 'id', 1: 'text', 2: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
bokmal = corpus
corpora.append(bokmal)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'btc'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.3)
btc = corpus
corpora.append(btc)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[3].to_tagged_string('ner'))


dataset = 'conll2esp'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, encoding="latin-1", in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.3)
conll2esp = corpus
corpora.append(conll2esp)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'conll2ned'
columns = {0: 'text', 1:'pos', 2: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, encoding="latin-1", in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
conll2ned = corpus
#corpora.append(conll2ned)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[10].to_tagged_string('ner'))


dataset = 'conll3deu'
columns = {0: 'text', 1: 'tkn', 2:'pos', 3: 'cnk', 4: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
conll3deu = corpus
corpora.append(conll3deu)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'conll3eng'
columns = {0: 'text', 1:'pos', 2: 'cnk', 3: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
conll3eng = corpus
corpora.append(conll3eng)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'daily'
columns = {0:'text', 1:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.1)
daily = corpus
corpora.append(daily) # comment out to exclude this corpus from training

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[1105].to_tagged_string('ner'))


dataset = 'estner'
columns = {0:'text', 1:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.1)
estner = corpus
corpora.append(estner)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'evalita'
columns = {0: 'text', 1:'pos', 2:'cde', 3:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, encoding="latin-1", in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
evalita = corpus
corpora.append(evalita)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'finer'
columns = {0: 'text', 1:'ner', 2:'mer'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
finer = corpus
corpora.append(finer)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[10].to_tagged_string('ner'))


dataset = 'hvgcontext'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, encoding="latin-1", in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
hvgcontext = corpus
corpora.append(hvgcontext)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[4].to_tagged_string('ner'))


dataset = 'msra'
columns = {0:'text', 1:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.1)
msra = corpus
corpora.append(msra) # comment out to exclude this corpus from training

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[1100].to_tagged_string('ner'))


dataset = 'nchlt'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.3)
nchlt = corpus
corpora.append(nchlt)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'nynorsk'
columns = {0: 'id', 1: 'text', 2: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
nynorsk = corpus
corpora.append(nynorsk)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[306].to_tagged_string('ner'))


dataset = 'persian'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.2)
persian = corpus
corpora.append(persian)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[3].to_tagged_string('ner'))


dataset = 'pucit'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.1)
pucit = corpus
corpora.append(pucit)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'ritter'
columns = {0: 'id', 1:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.9)
ritter = corpus
corpora.append(ritter)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'ronec'
columns = {0: 'id', 1:'text', 2:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.3)
ronec = corpus
corpora.append(ronec)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'sic'
columns = {0: 'id', 1:'text', 2:'word', 3: 'pos', 4:'sop', 5:'fnc', 6: 'ttl', 7:'abs', 8:'cpl', 9: 'zro', 10:'frk', 11:'ner', 12:'hsh'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.9)
sic = corpus
corpora.append(sic)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[12].to_tagged_string('ner'))


dataset = 'szeged'
columns = {0: 'text', 1: 'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, encoding="latin-1", in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.3)
szeged = corpus
corpora.append(szeged)

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[0].to_tagged_string('ner'))


dataset = 'weibo'
columns = {0:'text', 1:'ner'}
data_folder = f'{folder}/{dataset}'
corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=inplace)
if downsample == True: corpus = corpus.downsample(0.9)
weibo = corpus
corpora.append(weibo) # comment out to exclude this corpus from training

#print(corpus)
#print(corpus.obtain_statistics())
#print(corpus.make_tag_dictionary('ner'))
#print(corpus.train[100].to_tagged_string('ner'))


# #### XXVcorpus

# build XXVcorpus
xxv_corpus = MultiCorpus(corpora) # to exclude Chinese, remove 'daily', 'msra' & 'weibo'

tag_type = 'ner'
tag_dictionary = xxv_corpus.make_tag_dictionary(tag_type=tag_type)

#print(xxv_corpus)
print(tag_dictionary)


# #### Training

# initialize Flair embeddings for training XXVner-FM
embedding_types = [FlairEmbeddings('multi-forward'),
                   FlairEmbeddings('multi-backward')]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
# initialize sequence tagger and trainer
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, xxv_corpus)
    
# start training    
trainer.train(folder
              ,learning_rate=0.1 
              ,mini_batch_size=32 
              ,max_epochs=150)


# initialize embeddings for training XXVner-XLM
#embeddings = TransformerWordEmbeddings(model="xlm-roberta-large",
#                                        layers="all",
#                                        pooling_operation="first",
#                                        fine_tune=False,
#                                        use_scalar_mix=True)

# initialize sequence tagger and trainer
#tagger: SequenceTagger = SequenceTagger(hidden_size=256,
#                                        embeddings=embeddings,
#                                        tag_dictionary=tag_dictionary,
#                                        tag_type=tag_type,
#                                        use_crf=True) # use CRF?

#trainer: ModelTrainer = ModelTrainer(tagger, xxv_corpus)
    
# start training
#trainer.train(folder
#              ,learning_rate=0.1
#              ,mini_batch_size=32 # set smaller?
#              ,max_epochs=150)

