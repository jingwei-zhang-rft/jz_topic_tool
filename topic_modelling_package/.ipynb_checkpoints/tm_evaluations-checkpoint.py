import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from collections import Counter

# Gensim
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.models import CoherenceModel



def model_eval(model, number_of_topic, token_corpus, id2word, tfcount_corpus, coherence_metric = 'c_v', 
                  distance_metric = 'cosine', document_score_threshold = 0.5, topn_eval = 30, topn_output = 10):

    # Generate topics 
    topic_list = model2topics(model, number_of_topic, topn_terms = topn_eval)
    topic_list_saved = [','.join(i[:topn_output]) for i in topic_list]



    coherence_score_list = within_topic_coherence (topics = topic_list, 
                                                   token_corpus = token_corpus, 
                                                   tfcounter_corpus = tfcount_corpus,
                                                   id2word = id2word, 
                                                   topn_terms = topn_eval, 
                                                   coherence_metric = coherence_metric)

    distance_avg_list, distance_min_list = between_topic_distance(model = model, distance_metric = distance_metric)
    
    doc_dict = document_distribution(model = model, 
                                     tfcounter_corpus = tfcount_corpus, 
                                     confidence_level = document_score_threshold)



    topic_df = pd.DataFrame([list(range(number_of_topic+1))[1:],
                             coherence_score_list[:-1],
                             distance_avg_list[:-1],
                             distance_min_list[:-1], 
                             topic_list_saved]).transpose()
    topic_df.columns = ['topic_index','coherence','distance_avg','distance_min','top_terms']
    topic_df['document_distribution'] = topic_df['topic_index'].apply(lambda x: doc_dict[x] if x in list(doc_dict.keys()) else 0)
    topic_df['topic_number'] = number_of_topic
    topic_df = topic_df[['topic_number','topic_index','top_terms','coherence','distance_avg','distance_min','document_distribution']]



    model_result = {'topic_number': number_of_topic,
                    'coherence':coherence_score_list[-1],
                    'distance_avg':distance_avg_list[-1],
                    'distance_min':distance_min_list[-1],
                    'doc_distribution_std': topic_df.document_distribution.std(),
                    'doc_distribution_max': topic_df.document_distribution.max(),
                    'doc_unclassified_rate': 1- topic_df.document_distribution.sum()
                   }


    return model_result, topic_df
  




def model2topics (model, number_of_topics, topn_terms = 30):
    list_of_topics = []
    for i in list(range(number_of_topics)):
        topic_terms = []
        for term, score in model.show_topic(i, topn = topn_terms):
            topic_terms.append(term)
        list_of_topics.append(topic_terms)
    return list_of_topics


# model evaluation 



def within_topic_coherence (topics, token_corpus, tfcounter_corpus, id2word, topn_terms = 30, coherence_metric = 'c_v'):
    '''
    For ‘c_v’, ‘c_uci’ and ‘c_npmi’ texts should be provided (corpus isn’t needed); 
    For ‘u_mass’ corpus should be provided

    topn_terms: default of coherence model is 20

    return: a list of coherence score, each topic will get a coherence score, the last one is model score

    '''
    coherence_model = models.CoherenceModel(topics = topics, 
                                            texts = token_corpus, 
                                            corpus = tfcounter_corpus,
                                            dictionary = id2word, 
                                            coherence = coherence_metric,
                                            topn = topn_terms)

    model_coherence = coherence_model.get_coherence()
    topic_coherence = coherence_model.get_coherence_per_topic()
    topic_coherence.append(model_coherence)
    return topic_coherence
    




def between_topic_distance(model, distance_metric = 'cosine'):


    topic_term_matrix = model.get_topics()
    dist_out = pairwise_distances(topic_term_matrix, metric = distance_metric)
    model_distance_avg = (sum(np.unique(dist_out)) - 1)/len(np.unique(dist_out))
    topic_distance_avg = list(dist_out.sum(axis = 0)/(dist_out.shape[0] - 1))
    topic_distance_avg.append(model_distance_avg)

    np.fill_diagonal(dist_out, 1)
    topic_distance_min = list(dist_out.min(axis = 0))
    model_distance_min = np.mean(topic_distance_min)
    topic_distance_min.append(model_distance_min)
    return topic_distance_avg, topic_distance_min


def document_distribution(model, tfcounter_corpus, confidence_level = 0.5):
    doc_topic_score_list = list(model[tfcounter_corpus])

    dominant_topic_list = []
    for doc_score in doc_topic_score_list:
        for topic_score in doc_score:
            topic, score = topic_score
            if score > confidence_level:
                dominant_topic_list.append(topic + 1) 
    doc_dict = dict(Counter(dominant_topic_list))
    for key in list(doc_dict.keys()):
        doc_dict[key] = doc_dict[key]/len(doc_topic_score_list)

    return doc_dict



