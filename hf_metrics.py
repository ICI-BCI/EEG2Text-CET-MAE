import evaluate
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

class MyEvaluator:
    def __init__(self):
        print("Init MyEvaluator......")
        # self.rouge = evaluate.load("rouge")
        # self.rouge = load_metric("./rouge.py")
        self.rouge = Rouge()
        print("Load rouge")
        # self.wer = evaluate.load("wer")
        print("Load Word Error Rate")

    def score(self, predictions, references):
        # rouge_results = self.rouge.compute(predictions=predictions,
        #                                    references=references)
        rouge_results = self.rouge.get_scores(predictions, references, avg=True)
        print(rouge_results)
        # wer_score = self.wer.compute(predictions=predictions, references=references)

        score = {}
        score["rouge1"] = rouge_results["rouge-1"]
        score["rouge2"] = rouge_results["rouge-2"]
        score["rougel"] = rouge_results["rouge-l"]
        # score["wer"] = wer_score

        predictions_tokens = [prediction.split() for prediction in predictions]
        references_tokens = [[reference.split()] for reference in references]

        weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu(references_tokens, predictions_tokens, weights=weight)
            name = f"nltk_bleu_{len(list(weight))}"
            score[name] = corpus_bleu_score

        return score
