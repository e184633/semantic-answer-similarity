'''
All experiments (fingers crossed). 
Please note that for berttrained results, we had to make a local change in bert_score package, more precisely in utils.py function and add 
{'T-Systems-onsite/cross-en-de-roberta-sentence-transformer': 12} key-value pair to model2layers variable. 

Tests: Nah..
'''

import pandas as pd

from bert_score import score
from haystack.modeling.evaluation.squad_evaluation import compute_f1
from haystack.modeling.utils import initialize_device_settings
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import logging

BERTTRAINED = 'bert_score_prime'
BERT_SCORE = 'bert_score'
BERT_UNCASED = 'bert-base-uncased'
BIENCODER = 'bi_encoder'
SAS = 'sas'
F1_SCORE = 'f1'


def main():
    device, n_gpu = initialize_device_settings(use_cuda=False)
    logging.set_verbosity_error()
    squad = pd.read_csv('data/data/SQuAD_SAS.csv')
    nq_open = pd.read_csv('data/data/NQ-open_SAS.csv')
    german_quad = pd.read_csv('data/data/GermanQuAD_SAS.csv')

    # 1. Bi-Encoder approach: all dataset experiments together
    model_type_trained = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    model = SentenceTransformer(model_type_trained)
    for data, name in zip([squad, nq_open, german_quad], ('squad', 'nq-open', 'german_squad')):
        scores_bi_encoder = data.copy()
        embeddings1 = model.encode(data.answer1, convert_to_tensor=True)
        embeddings2 = model.encode(data.answer2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        scores_bi_encoder[BIENCODER] = pd.Series()
        for i in range(len(data['answer1'])):
            scores_bi_encoder[BIENCODER][i] = cosine_scores[i][i].item()

        scores_bi_encoder[F1_SCORE] = scores_bi_encoder.apply(lambda x: 
            compute_f1(str(x.answer1), str(x.answer2)), axis=1)
        scores_bi_encoder.to_csv(f'{name}_bi_encoder_score.csv')

    # 2. Cross-Encoder approach

    ## SQuAD
    english_model_cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')
    series_squad = pd.Series(
        [english_model_cross_encoder.predict([
            str(squad.answer1.values[i]),
            str(squad.answer2.values[i])],
            show_progress_bar=False)
         for i in range(len(squad))])
    squad[SAS] = series_squad

    ## German QuAD
    german_model_cross_encoder = CrossEncoder('deepset/gbert-large-sts')

    series_german = pd.Series(
        [german_model_cross_encoder.predict([str(german_quad.answer1.values[i]),
                                      str(german_quad.answer2.values[i])],
                                     show_progress_bar=False)
                                     for i in range(len(german_quad))])
    german_quad[SAS] = series_german
    german_quad.to_csv('data/table/german_quad_sas.csv')

    ## NQ-open
    series_nq_open = pd.Series(
        [english_model_cross_encoder.predict([
            str(nq_open.answer1.values[i]),
            str(nq_open.answer2.values[i])],
            show_progress_bar=False)
         for i in range(len(nq_open))])

    nq_open[SAS] = series_nq_open
    nq_open.to_csv('data/table/nq_open_sas.csv')

    # BERTScore vanilla approach

    ## SQUAD 
    english_bert_score = []
    for i in range(len(squad)):
        _, _, bertscore = score(
            [squad.answer1[i]],
            [squad.answer2[i]],
            model_type=BERT_UNCASED,
            num_layers=2)
        english_bert_score.append(bertscore.item())
    squad[BERT_SCORE] = english_bert_score

    ## German QuAD
    german_series_bert_score = []
    for i in range(len(german_quad)):
        _, _, bertscore = score(
            [str(german_quad.answer1[i])],
            [str(german_quad.answer2[i])],
            model_type='deepset/gelectra-base',
            num_layers=2)
        german_series_bert_score.append(bertscore.item())
    german_quad[BERT_SCORE] = german_series_bert_score

    ## NQ-open
    for i in range(len(nq_open)):
        _, _, bertscore = score(
            [str(nq_open.answer1[i])],
            [str(nq_open.answer2[i])],
            model_type=BERT_UNCASED,
            num_layers=2)
        nq_open_series_bert_score.append(bertscore.item())
    nq_open[BERT_SCORE] = nq_open_series_bert_score
    nq_open.to_csv('nq_open_bert.csv')

    # BERTScore Trained

    ## SQuAD
    bert_score_prime = []
    for i in range(len(squad)):
        _, _, bertscore = score(
            cands=[squad.answer1[i]],
            refs=[squad.answer2[i]],
            model_type=model_type_trained)
        bert_score_prime.append(bertscore.item())
    squad[BERTTRAINED] = bert_score_prime    
    squad.to_csv('data/table/squad_table_results.csv')

    ## German QuAD
    german_quad_bert_trained = []
    for i in range(len(german_quad)):
        _, _, bertscore = score(
            [str(german_quad.answer1[i])],
            [str(german_quad.answer2[i])],
            model_type=model_type_trained,
            verbose=False)
        german_quad_bert_trained.append(bertscore.item())
    german_quad[BERTTRAINED] = german_quad_bert_trained
    german_quad.to_csv('german_quad_bert_trained.csv')

    ## NQ-open
    nq_open_bert_trained = []
    for i in range(len(nq_open)):
        _, _, bertscore = score(
            [str(nq_open.answer1[i])],
            [str(nq_open.answer2[i])],
            model_type=BERT_UNCASED)
        nq_open_bert_trained.append(bertscore.item())
    nq_open[BERTTRAINED] = nq_open_series_bert_score
    nq_open.to_csv('nq_open_bert_trained.csv')
