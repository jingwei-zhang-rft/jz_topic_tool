import pandas as pd
import numpy as np

# Gensim
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.models import CoherenceModel


# n-gram builder - list of tokens 
def ngram_tokenizer (texts, min_count=5, threshold=100, n_gram = 2):
    '''
    this function detetct potential phrases (e.g. bigram and trigram) in the token list 
    Input: a list of list of natural tokens (split by white space)
    return: a list of lists with n-gram tokens 
    '''
    list_of_lists_tokens = [i.split() for i in texts]
    if n_gram == 1:
        return list_of_lists_tokens
    else:
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(list_of_lists_tokens, min_count=min_count, threshold=threshold) # higher threshold fewer phrases.   
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        if n_gram == 3:
            trigram = gensim.models.Phrases(bigram[list_of_lists_tokens], threshold=threshold)  
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in list_of_lists_tokens]
        elif n_gram == 2:
            return [bigram_mod[doc] for doc in list_of_lists_tokens]
        else:
            print('Warning: N-GRAM SETTING IS INALID. PLEASE RESET IT TO 1, 2 OR 3!')


# vector features 
def tf_counter(list_of_lists_tokens, id2word, counter_type = 'bow'):

	
    bow_corpus = [id2word.doc2bow(text) for text in list_of_lists_tokens]

    if counter_type == 'tfidf':
        tfidf_model = models.TfidfModel(bow_corpus)
        tfidf_corpus = tfidf_model[bow_corpus]
        return list(tfidf_corpus)
    elif counter_type == 'bow':
        return bow_corpus
    else:
        print('Warning: THE COUNTER TYPE IS NOT SUPPORTED!')


# models
# Build LDA model

def model_builder (tf_corpus, id2word, number_of_topics, model_name = 'lda'):

    if model_name == 'lda':

        model = models.ldamodel.LdaModel(corpus = tf_corpus,
                                         id2word=id2word,
                                         num_topics=number_of_topics, 
                                         random_state=100,
                                         update_every=1,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         per_word_topics=False)

    elif model_name == 'lsi':

        model = models.LsiModel(corpus = tf_corpus,
                                id2word =id2word,
                                num_topics = number_of_topics)

    else:

        print('MODEL NAME CANNOT BE RECOGNISED!')


    return model
