import pandas as pd
import numpy as np
import gensim.corpora as corpora
import tm_models as tm_model
import tm_evaluations as tm_eval
from importlib import reload
reload(tm_model)
reload(tm_eval)


class PipelineTopicModeling:
      
    def __init__(self, text_data, n_gram = 2, tfcounter_type = 'bow', model_type = 'lda', topic_range_set = [None,None,1] ):
        '''
        text_data: list of strings (doc) of target documents 
        n_gram_set: 1 - unigram; 2 - bigram; 3 - trigram 
        tfcounter_type: term freqeuncy counter - 'bow' or 'tfidf' 
        model_type: 'lda','lda_multicore','lda_mallet','lsi'
        topic_range_set: a list of three int. representing min and max topic numbers and the stepsize



        '''
        # Parameters
        # tokenizer setting
        self.n_gram = n_gram
        # term frequency setting
        self.tfcounter_type = tfcounter_type
        # model setting
        self.model_type = model_type
        # number_of_topics 
        self.topics_range = range(topic_range_set[0], topic_range_set[1] +1, topic_range_set[2])


        self.string_corpus = text_data
        self.token_corpus = tm_model.ngram_tokenizer(texts = self.string_corpus,n_gram = self.n_gram)

        self.id2word = corpora.Dictionary(self.token_corpus)
        self.tf_corpus = tm_model.tf_counter(self.token_corpus, self.id2word, self.tfcounter_type)

        self.model_saved = {}
        for i in self.topics_range:
            self.model_saved[i] = tm_model.model_builder(self.tf_corpus, self.id2word, i, model_name = self.model_type)




    def get_evaluation_result(self, 
                              coherence_metric = 'c_v', 
                              distance_metric = 'cosine', 
                              document_score_threshold = 0.5, 
                              topn_eval = 30, 
                              topn_output = 10):

        '''



        '''

        model_result_list = []
        topic_result_df = pd.DataFrame()
        for topic_number in self.topics_range:

            model_result, topic_df = tm_eval.model_eval(model = self.model_saved[topic_number], 
                                                        number_of_topic = topic_number,
                                                        token_corpus = self.token_corpus,
                                                        id2word = self.id2word,
                                                        tfcount_corpus = self.tf_corpus,
                                                        coherence_metric = coherence_metric, 
                                                        distance_metric = distance_metric, 
                                                        document_score_threshold = document_score_threshold, 
                                                        topn_eval = topn_eval, 
                                                        topn_output = topn_output)

            model_result_list.append(model_result)
            topic_result_df = topic_result_df.append(topic_df)
            model_result_df = pd.DataFrame(model_result_list)

            model_result_df = model_result_df[['topic_number','coherence', 'distance_avg','distance_min', 'doc_distribution_max', 'doc_distribution_std','doc_unclassified_rate']]
        return model_result_df, topic_result_df

        















 

