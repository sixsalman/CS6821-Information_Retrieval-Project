import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import multiprocessing
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.measures import *

nltk.download('stopwords')
nltk.download('punkt')
# es_stemer = SnowballStemmer('spanish')
# es_stopwords = set(stopwords.words('spanish'))
result={}

def IndexCreator(dataset,filename, tokeniser="UTFTokeniser",stemmer=None,stopwords=None):
    if not os.path.exists(filename):
        indexer = pt.IterDictIndexer(filename, 
            stemmer=stemmer, stopwords=stopwords, # Removes the default PorterStemmer (English)
            tokeniser=tokeniser) # Replaces the default EnglishTokeniser, which makes assumptions specific to English
        index_ = indexer.index(dataset.get_corpus_iter())
    else:
        index_ = pt.IndexRef.of(filename)
    return index_


def custom_preprocess(text):
    toks = word_tokenize(text) # tokenize
    toks = [t for t in toks if t.lower() not in custom_stopwords] # remove stop words
    toks = [custom_stemmer.stem(t) for t in toks] # stem
    return ' '.join(toks) # combine toks back into a string



# Custom Preprocessing
# NB: This custom pre-processing ends up being considerably slower than using Terrier's built-in processor,
# so we use the multiprocessing package to parallelize (400 docs/s vs 2000 docs/s).
def map_doc(document,custom_preprocess=custom_preprocess):
    # this function replaces the document text with the version that uses our custom pre-processing
    return {
        'docno': document['docno'],
        'text': custom_preprocess(document['text'])
    }

def CustomIndexCreator(filename, tokeniser="UTFTokeniser",stemmer=None,stopwords=None,mapper=map_doc):

    if not os.path.exists(filename):
        indexer = pt.IterDictIndexer(filename, 
            stemmer=None, stopwords=None,  # Disable the default PorterStemmer (English)
            tokeniser=tokeniser) # Replaces the default EnglishTokeniser, which makes assumptions specific to English
        with multiprocessing.Pool() as pool:
            index_custom = indexer.index(pool.imap(mapper, dataset.get_corpus_iter()))
    else:
        index_custom = pt.IndexRef.of(filename)
    return index_custom



config={
1:
    {
        "dataset":"irds:wikir/en59k/test",
        "lang":"english",
        "tokeniser":"",
        "stemmer":"EnglishSnowballStemmer",
        "stopwords":None
    }
}

for i in tqdm(range(1,2)):
    print(i)
    config_data=config[i]
    result[config_data["lang"]]={}

    custom_stemmer=None
    custom_stopwords=None

    dataset = pt.get_dataset(config_data["dataset"])
    custom_stemmer = SnowballStemmer(config_data["lang"])
    custom_stopwords = set(stopwords.words(config_data["lang"]))
    def custom_preprocess(text):
        toks = word_tokenize(text) # tokenize
        toks = [t for t in toks if t.lower() not in custom_stopwords] # remove stop words
        toks = [custom_stemmer.stem(t) for t in toks] # stem
        return ' '.join(toks) # combine toks back into a string

    def map_doc(document,custom_preprocess=custom_preprocess):
        # this function replaces the document text with the version that uses our custom pre-processing
        return {
            'docno': document['docno'],
            'text': custom_preprocess(document['text'])
        }

    index_nostem = IndexCreator(dataset=dataset,filename=f'./wikir-{config_data["lang"]}-nostem',
                                tokeniser="UTFTokeniser",stemmer=None,stopwords=None)
    index_stem=IndexCreator(dataset=dataset,filename=f'./wikir-{config_data["lang"]}-stem', 
                            tokeniser="UTFTokeniser",stemmer=config_data["stemmer"],stopwords=None)
    index_custom = CustomIndexCreator(filename=f'./wikir-{config_data["lang"]}-custom_stem', tokeniser="UTFTokeniser",stemmer=None,stopwords=None,mapper=map_doc)




    bm25_nostem = pt.BatchRetrieve(index_nostem, wmodel='BM25')
    bm25_stem = pt.BatchRetrieve(index_stem, wmodel='BM25')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    bm25_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='BM25')


    tfidf_nostem = pt.BatchRetrieve(index_nostem, wmodel='TF_IDF')
    tfidf_stem = pt.BatchRetrieve(index_stem, wmodel='TF_IDF')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    tfidf_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='TF_IDF')



    bb2_nostem = pt.BatchRetrieve(index_nostem, wmodel='BB2')
    bb2_stem = pt.BatchRetrieve(index_stem, wmodel='BB2')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    bb2_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='BB2')



    
    
st.title('Information retreival Fucntion')

# Using the "with" syntax
with st.form(key='my_form'):
	text_input = st.text_input(label='Enter some text')
	submit_button = st.form_submit_button(label='Submit')
st.write(text_input)
data=bm25_custom.search(str(text_input))
st.dataframe(data)import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import multiprocessing
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.measures import *

nltk.download('stopwords')
nltk.download('punkt')
# es_stemer = SnowballStemmer('spanish')
# es_stopwords = set(stopwords.words('spanish'))
result={}

def IndexCreator(dataset,filename, tokeniser="UTFTokeniser",stemmer=None,stopwords=None):
    if not os.path.exists(filename):
        indexer = pt.IterDictIndexer(filename, 
            stemmer=stemmer, stopwords=stopwords, # Removes the default PorterStemmer (English)
            tokeniser=tokeniser) # Replaces the default EnglishTokeniser, which makes assumptions specific to English
        index_ = indexer.index(dataset.get_corpus_iter())
    else:
        index_ = pt.IndexRef.of(filename)
    return index_


def custom_preprocess(text):
    toks = word_tokenize(text) # tokenize
    toks = [t for t in toks if t.lower() not in custom_stopwords] # remove stop words
    toks = [custom_stemmer.stem(t) for t in toks] # stem
    return ' '.join(toks) # combine toks back into a string



# Custom Preprocessing
# NB: This custom pre-processing ends up being considerably slower than using Terrier's built-in processor,
# so we use the multiprocessing package to parallelize (400 docs/s vs 2000 docs/s).
def map_doc(document,custom_preprocess=custom_preprocess):
    # this function replaces the document text with the version that uses our custom pre-processing
    return {
        'docno': document['docno'],
        'text': custom_preprocess(document['text'])
    }

def CustomIndexCreator(filename, tokeniser="UTFTokeniser",stemmer=None,stopwords=None,mapper=map_doc):

    if not os.path.exists(filename):
        indexer = pt.IterDictIndexer(filename, 
            stemmer=None, stopwords=None,  # Disable the default PorterStemmer (English)
            tokeniser=tokeniser) # Replaces the default EnglishTokeniser, which makes assumptions specific to English
        with multiprocessing.Pool() as pool:
            index_custom = indexer.index(pool.imap(mapper, dataset.get_corpus_iter()))
    else:
        index_custom = pt.IndexRef.of(filename)
    return index_custom



config={
1:
    {
        "dataset":"irds:wikir/en59k/test",
        "lang":"english",
        "tokeniser":"",
        "stemmer":"EnglishSnowballStemmer",
        "stopwords":None
    }
}

for i in tqdm(range(1,2)):
    print(i)
    config_data=config[i]
    result[config_data["lang"]]={}

    custom_stemmer=None
    custom_stopwords=None

    dataset = pt.get_dataset(config_data["dataset"])
    custom_stemmer = SnowballStemmer(config_data["lang"])
    custom_stopwords = set(stopwords.words(config_data["lang"]))
    def custom_preprocess(text):
        toks = word_tokenize(text) # tokenize
        toks = [t for t in toks if t.lower() not in custom_stopwords] # remove stop words
        toks = [custom_stemmer.stem(t) for t in toks] # stem
        return ' '.join(toks) # combine toks back into a string

    def map_doc(document,custom_preprocess=custom_preprocess):
        # this function replaces the document text with the version that uses our custom pre-processing
        return {
            'docno': document['docno'],
            'text': custom_preprocess(document['text'])
        }

    index_nostem = IndexCreator(dataset=dataset,filename=f'./wikir-{config_data["lang"]}-nostem',
                                tokeniser="UTFTokeniser",stemmer=None,stopwords=None)
    index_stem=IndexCreator(dataset=dataset,filename=f'./wikir-{config_data["lang"]}-stem', 
                            tokeniser="UTFTokeniser",stemmer=config_data["stemmer"],stopwords=None)
    index_custom = CustomIndexCreator(filename=f'./wikir-{config_data["lang"]}-custom_stem', tokeniser="UTFTokeniser",stemmer=None,stopwords=None,mapper=map_doc)




    bm25_nostem = pt.BatchRetrieve(index_nostem, wmodel='BM25')
    bm25_stem = pt.BatchRetrieve(index_stem, wmodel='BM25')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    bm25_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='BM25')


    tfidf_nostem = pt.BatchRetrieve(index_nostem, wmodel='TF_IDF')
    tfidf_stem = pt.BatchRetrieve(index_stem, wmodel='TF_IDF')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    tfidf_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='TF_IDF')



    bb2_nostem = pt.BatchRetrieve(index_nostem, wmodel='BB2')
    bb2_stem = pt.BatchRetrieve(index_stem, wmodel='BB2')
    # to apply the es_preprocess function to the query text, use a pt.apply.query transformer
    bb2_custom = pt.apply.query(lambda row: custom_preprocess(row.query)) >> pt.BatchRetrieve(index_custom, wmodel='BB2')



    
    
st.title('Information retreival Fucntion')

# Using the "with" syntax
with st.form(key='my_form'):
	text_input = st.text_input(label='Enter some text')
	submit_button = st.form_submit_button(label='Submit')
st.write(text_input)
data=bm25_custom.search(str(text_input))
st.dataframe(data)
