import json
import numpy as np
from rouge_chinese import Rouge
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.nist_score import corpus_nist, sentence_nist
from nltk.translate.meteor_score import meteor_score
import argparse
from bert_score import score
from pycocoevalcap.cider.cider import Cider
import pandas as pd
import jieba

def Rouge_Score(hypothesis, reference):
    rouger = Rouge()
    hypothesis_cut = [' '.join(jieba.cut(item)) for item in hypothesis]
    reference_cut = [' '.join(jieba.cut(item)) for item in reference]
    scores = rouger.get_scores(hypothesis_cut, reference_cut)
    arr_1 = np.array([[item['rouge-1']['r'], item['rouge-1']['p'], item['rouge-1']['f']] for item in scores])
    avg_r_1, avg_p_1, avg_f_1 = np.mean(arr_1[:, 0]), np.mean(arr_1[:, 1]), np.mean(arr_1[:, 2])
    arr_2 = np.array([[item['rouge-2']['r'], item['rouge-2']['p'], item['rouge-2']['f']] for item in scores])
    avg_r_2, avg_p_2, avg_f_2 = np.mean(arr_2[:, 0]), np.mean(arr_2[:, 1]), np.mean(arr_2[:, 2])
    arr_l = np.array([[item['rouge-l']['r'], item['rouge-l']['p'], item['rouge-l']['f']] for item in scores])
    avg_r_l, avg_p_l, avg_f_l = np.mean(arr_l[:, 0]), np.mean(arr_l[:, 1]), np.mean(arr_l[:, 2])
    avg_scores = {
        'rouge-1': {'r': avg_r_1, 'p': avg_p_1, 'f': avg_f_1},
        'rouge-2': {'r': avg_r_2, 'p': avg_p_2, 'f': avg_f_2}, 
        'rouge-l': {'r': avg_r_l, 'p': avg_p_l, 'f': avg_f_l}
        }
    return avg_scores, arr_l[:, 2]

def Bleu_Scores(candidates, references):
    references  = np.array(references).reshape(-1, 1)
    references_tokens = [[list(jieba.cut(s)) for s in sentences] for sentences in references]
    candidates_tokens = [list(jieba.cut(sentence)) for sentence in candidates]
    bleu_scores = [corpus_bleu(references_tokens, candidates_tokens, weights=(n,)) for n in range(1, 5)]
    bleu_1_scores = [sentence_bleu(r, z, weights=(1,)) for r, z in zip(references_tokens, candidates_tokens)]
    ave_scores = {}
    for n, score in enumerate(bleu_scores, start=1):
       ave_scores[f"BLEU-{n}"] = score
    return ave_scores, bleu_1_scores

def Nist_Scores(candidates, references):
    references  = np.array(references).reshape(-1, 1)
    references_tokens = [[list(jieba.cut(s)) for s in sentences] for sentences in references]
    candidates_tokens = [list(jieba.cut(sentence)) for sentence in candidates]

    nist_5 = corpus_nist(references_tokens, candidates_tokens, n=5)

    nist_5_scores = [sentence_nist(r, z, n=min(len(r), 5)) for r, z in zip(references_tokens, candidates_tokens)]

    return nist_5 , nist_5_scores

def Meteor_Scores(candidates, references):
    references  = np.array(references).reshape(-1, 1)
    references_tokens = [[list(jieba.cut(s)) for s in sentences] for sentences in references]
    candidates_tokens = [list(jieba.cut(sentence)) for sentence in candidates]

    meteor_scores = [meteor_score(ref, candidate) for ref, candidate in zip(references_tokens, candidates_tokens)]
    return np.mean(meteor_scores).item(), meteor_scores

def Bert_Score(candidates, references):

    P, R, F1 = score(candidates, references, model_type="roberta-large", verbose=False)

    return {"BERT-P:": float(np.mean(P.numpy())), "BERT-R:": float(np.mean(R.numpy())), "BERT-F1:": float(np.mean(F1.numpy()))}, F1.numpy()


def Cider_Score(candidates, references):

    candidates = {i: [' '.join(jieba.cut(sentence))] for i, sentence in enumerate(candidates)}
    references = {i: [' '.join(jieba.cut(sentence))] for i, sentence in enumerate(references)}
    cider_scorer = Cider()
    # 计算CIDEr分数
    ave_score, scores = cider_scorer.compute_score(candidates, references)
    return ave_score.item(), scores


# $ python .\Scores_cn.py --answers "./Answers_cn/ChatGLM2-6B-langchain-answer.jsonl" --answers_key 'LLM-A' --result "./Scores_cn/ChatGLM2-6B_scores"
# $ python .\Scores_cn.py --answers "./Answers_cn/ChatGLM-6B-langchain-answer.jsonl" --answers_key 'LLM-A' --result "./Scores_cn/ChatGLM1-6B_scores"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against key labels")

    parser.add_argument("--label",default="./Answers_cn/QA_cn.jsonl", type=str, help="Path to label file")
    parser.add_argument("--answers", type=str, help="Path to answers file")
    parser.add_argument("--answers_key", type=str, help="Key to answers file")
    parser.add_argument("--result", type=str, help="Path to result file(no suffix)")

    args = parser.parse_args()

    label_path = args.label
    answers_path = args.answers
    answers_key = args.answers_key
    result_path = args.result
    Ques, Labels, Answers = [], [], []        
    with open(label_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            Ques.append(data['Q'])
            Labels.append(data['A'])
    with open(answers_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            Answers.append(data[answers_key])
    assert len(Answers) == len(Labels), f"Lengths of answers != Lengths of labels."
    score_details = pd.DataFrame()
    score_details['Q'], score_details['A'], score_details['LLM_A'] = Ques, Labels, Answers
    result = {}

    result['ROUGE'], score_details['ROUGE-L'] = Rouge_Score(Answers, Labels)
    result['BLEU'], score_details['BLEU-1'] = Bleu_Scores(Answers, Labels) 
    result['NIST'], score_details['NIST-5'] = Nist_Scores(Answers, Labels)
    result['METEOR'], score_details['METEOR']  = Meteor_Scores(Answers, Labels)
    # result['BERT'], score_details['BERT']  = Bert_Score(Answers, Labels)
    result['CIDEr'], score_details['CIDEr']  = Cider_Score(Answers, Labels)
        
    with open(f'{result_path}.json', "w") as json_file:
        json.dump(result, json_file)
    
    score_details.to_excel(f'{result_path}.xlsx', index=False)


