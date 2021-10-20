from flask import Flask 
import pandas as pd
from all_in_one_topic_modelling import PipelineTopicModeling 
# 
app = Flask(__name__)


test_df = pd.read_csv('trump_trade_corpus.csv')

# home endpoint
@app.route('/') 
def get_topics():
    
    pltm_object = PipelineTopicModeling(text_data = test_df, 
                                    n_gram=2,
                                    tfcounter_type='bow',
                                    model_type='lda',
                                    topic_range_set = [4,10,1])

    model_eval,topic_df = pltm_object.get_evaluation_result(coherence_metric = 'c_v', 
                                                          distance_metric = 'cosine', 
                                                          document_score_threshold = 0.5, 
                                                          topn_eval = 20, 
                                                          topn_output = 10)

    return model_eval.to_json()


# home endpoint 123
@app.route('/test') 
def test():
    return "ooo"
# host must be added to run it with docker 
if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")



