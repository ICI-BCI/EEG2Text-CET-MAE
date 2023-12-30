# import evaluate
from nltk.translate.bleu_score import corpus_bleu
from evaluate_moudle.rouge import Rouge
from evaluate_moudle.wer import WER
# 第二个评价指标
from rouge import Rouge as Rouge2

class MyEvaluator_for_test:
    def __init__(self):
        print("Init MyEvaluator......")
        # self.rouge = evaluate.load("rouge")
        self.rouge = Rouge()
        self.rouge2 = Rouge2()
        print("Load rouge")
        # self.wer = evaluate.load("wer")
        self.wer = WER()
        print("Load Word Error Rate")

    def score(self, predictions, references):
        # Zeng(TNSRE,2023)
        rouge_results = self.rouge.compute(predictions=predictions,
                                           references=references)
        wer_score = self.wer.compute(predictions=predictions, references=references)

        # Wang(AAAI,2022)
        rouge_scores = self.rouge2.get_scores(predictions, references, avg=True) # 这里返回的就是一个字典
        # print(rouge_scores)

        all_score = {}
        all_score["rouge1"] = rouge_results["rouge1"]
        all_score["rouge2"] = rouge_results["rouge2"]
        all_score["rougel"] = rouge_results["rougeL"]
        all_score["wer"] = wer_score

        predictions_tokens = [prediction.split() for prediction in predictions]
        references_tokens = [[reference.split()] for reference in references]

        weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu(references_tokens, predictions_tokens, weights=weight)
            name = f"nltk_bleu_{len(list(weight))}"
            all_score[name] = corpus_bleu_score

        return all_score,rouge_scores



class MyEvaluator:
    def __init__(self):
        print("Init MyEvaluator......")
        self.rouge = Rouge()
        print("Load rouge")
        self.wer = WER()
        print("Load Word Error Rate")

    def score(self, predictions, references):
        rouge_results = self.rouge.compute(predictions=predictions,
                                           references=references)
        wer_score = self.wer.compute(predictions=predictions, references=references)

        score = {}
        score["rouge1"] = rouge_results["rouge1"]
        score["rouge2"] = rouge_results["rouge2"]
        score["rougel"] = rouge_results["rougeL"]
        score["wer"] = wer_score

        predictions_tokens = [prediction.split() for prediction in predictions]
        references_tokens = [[reference.split()] for reference in references]

        weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu(references_tokens, predictions_tokens, weights=weight)
            name = f"nltk_bleu_{len(list(weight))}"
            score[name] = corpus_bleu_score

        return score


















































































































































































































































































