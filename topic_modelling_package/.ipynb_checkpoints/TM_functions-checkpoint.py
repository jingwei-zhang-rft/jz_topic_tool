import pandas as pd
import numpy as np

# Gensim
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.models import CoherenceModel


class PipelineTopicModeling:
      
    def __init__(self, text_data, n_gram_set = 2, corpus_type = 'bow', model_list = ['lda'], number_of_topics = None ):
        '''
        text_data: list of strings (doc) of target documents 
        n_gram_set: 1 - unigram; 2 - bigram; 3 - trigram 
        corpus_type: 'bow' or 'tfidf' 
        model-list: 'lda','lda_mallet'
        number of topics: 
           - int: 
           - list(range())



        '''

        self.string_corpus = text_data
        self.token_corpus = [i.split() for i in self.string_corpus.tolist()]

        self.id2word = corpora.Dictionary(self.token_corpus)

        self.n_gram = n_gram_set
        self.corpus_type = corpus_type
        self.model_dict = None
         
        self.token_corpus_ngram = n_gram_builder(self.token_corpus,n_gram = self.n_gram)

        self.freq_corpus = frequency_corpus_builder(self.token_corpus_ngram, self.id2word, self.corpus_type)


        # Evaluation 
        self.topn_terms = topn_terms

    

    # def plot_results():





    def tuning_result(self, model, topic_number, coherence_metric = 'c_v', topn_terms = 30,topn_saved = 10):
       
        topic_list = model2topics (model, topn_terms = topn_terms)
        topic_list_saved = [','.join(i[:topn_saved]) for i in topic_list]
        topic_list_saved.append(None)



        coherence_score_list = within_topic_coherence (topic_list, 
                                                       list_of_lists_tokens = self.token_corpus_ngram, 
                                                       corpus = self.freq_corpus,
                                                       id2word = self.id2word, 
                                                       topn_terms = topn_terms, 
                                                       coherence_metric = 'c_v')

        distance_score_list = between_topic_distance(model, distance_metric = 'cosine')

        return pd.dataFrame([coherence_score_list,distance_score_list,topic_list_saved])

        












# n-gram builder - list of tokens 
def n_gram_builder (list_of_lists_tokens, min_count=5, threshold=100, n_gram = 2):
    '''
    this function detetct potential phrases (e.g. bigram and trigram) in the token list 
    Input: a list of list of natural tokens (split by white space)
    return: a list of lists with n-gram tokens 
    '''
    if n_gram == 1:
        return list_of_lists_tokens
    else:
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(list_of_lists, min_count=min_count, threshold=threshold) # higher threshold fewer phrases.   
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        if n_gram == 3:
           trigram = gensim.models.Phrases(bigram[list_of_lists], threshold=threshold)  
           trigram_mod = gensim.models.phrases.Phraser(trigram)
           return [trigram_mod[bigram_mod[doc]] for doc in list_of_lists_tokens]
        elif n_gram == 2:
           return [bigram_mod[doc] for doc in list_of_lists_tokens]
        else:
            print('N-GRAM SETTING IS INALID. PLEASE RESET IT TO 1, 2 OR 3!')


# vector features 
def frequency_corpus_builder(list_of_lists_tokens, id2word, freq_type = 'bow'):
    bow_corpus = [id2word.doc2bow(list_of_lists_tokens) for text in list_of_lists_tokens]

    if freq_type == 'tfidf':
       tfidf_model = models.TfidfModel(bow_corpus)
       tfidf_corpus = tfidf_model[corpus]
       return tfidf_corpus
    else:
       return bow_corpus


# models
# Build LDA model

def tm_mdoels (corpus, id2word, number_of_topics, model_selection, model_name = 'lda'):

    if mdoel_name == 'lda':

        model = models.ldamodel.LdaModel(corpus = corpus,
                                             id2word=id2word,
                                             num_topics=number_of_topics, 
                                             random_state=100,
                                             update_every=1,
                                             chunksize=100,
                                             passes=10,
                                             alpha='auto',
                                             per_word_topics=False)

    elif model_name == 'lsi':

        model = models.LsiModel(corpus = corpus,
                                id2word =id2word,
                                num_topics = number_of_topics)

    else:

        print('MODEL NAME CANNOT BE RECOGNISED!')


    return model


def model2topics (model, topn_terms = 30):
    list_of_topics = []
    for i in list(range(5)):
        topic_terms = []
        for term, score in model.show_topic(i, topn = topn_terms):
            topic_terms.append(term)
        list_of_topics.append(topic_terms)
    return list_of_topics


# model evaluation 



def within_topic_coherence (topics, list_of_lists_tokens, corpus, id2word, topn_terms = 30, coherence_metric = 'c_v'):
    '''
    For ‘c_v’, ‘c_uci’ and ‘c_npmi’ texts should be provided (corpus isn’t needed); 
    For ‘u_mass’ corpus should be provided

    topn_terms: default of coherence model is 20

    return: a list of coherence score, each topic will get a coherence score, the last one is model score

    '''
    coherence_model = models.CoherenceModel(topics = topics, 
                                     texts = list_of_lists_tokens, 
                                     corpus = corpus,
                                     dictionary = id2word, 
                                     coherence = coherence_metric,
                                     topn = topn_terms

                                         )
    model_coherence = coherence_model.get_coherence()
    topic_coherence = coherence_model.get_coherence_per_topic(segmented_topics=None, with_std=False, with_support=False)
    return topic_coherence.append(model_coherence)
    




def between_topic_distance(model, distance_metric = 'cosine'):
    topic_term_matrix = model.get_topics()
    dist_out = 1-pairwise_distances(topic_term_matrix, metric = distance_metric)
    model_distance = (sum(np.unique(dist_out)) -1)/len(np.unique(dist_out))
    np.fill_diagonal(dist_out, 0)
    topic_distance = list(dist_out.max(axis = 0))
    return topic_distance.append(model_distance)






