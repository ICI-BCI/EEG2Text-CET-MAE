import os
import argparse
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, \
    BartForSequenceClassification, BertTokenizer, BertConfig, \
    BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, \
    T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config, \
    MvpTokenizer, MvpConfig, MvpForConditionalGeneration
from dataset import EEG_dataset_add_sentence_clip as EEG_dataset
from model import BrainTranslator,BrainMVPos,BrainMultiStreamMVP,BrainMVP
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from hf_metrics_vanilla import MyEvaluator
from optim_new import *
from rouge import Rouge
"""
follow by the article "Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification"
"""

"""
task1+task2+taskNRv2 : train_sample : 670*16 = 10710
"""


checkpoint_best = './checkpoints/decoding/best/'

if not os.path.exists(checkpoint_best):
    os.makedirs(checkpoint_best)

checkpoint_last = './checkpoints/decoding/last/'
if not os.path.exists(checkpoint_last):
    os.makedirs(checkpoint_last)

def eval_model(dataloaders, device, tokenizer, model, output_all_results_path='./results/temp.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    my_evaluator = MyEvaluator()
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    # Iterate over data.
    sample_count = 0

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path, 'w') as f:
        f.write(eeg2text_checkpoint)
        f.write('\n')
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders['test']):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_string)
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100

            # forward
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                    target_ids_batch)

            """calculate loss"""
            # logits = seq2seqLMoutput.logits # 8*48*50265
            # logits = logits.permute(0,2,1) # 8*50265*48

            # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
            # NOTE: my criterion not used
            loss = seq2seqLMoutput.loss  # use the BART language modeling loss

            # get predicted tokens
            # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size())
            # logits: 1*56*50265
            logits = seq2seqLMoutput.logits  # 1*56*50265

            # logits = logits.permute(0,2,1)
            # print('permuted logits size:', logits.size())

            # probs: 56*50265
            probs = logits[0].softmax(dim=1)
            # print('probs size:', probs.size())
            values, predictions = probs.topk(1)  # 从 词库表中挑选一个概率最大的词作为输出，torch.size([56,1])
            # print('predictions before squeeze:',predictions.size())
            predictions = torch.squeeze(predictions)  # predictions: (56,)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break

            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            sample_count += 1
            # statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss
            # print('[DEBUG]loss:',loss.item())
            # print('#################################')

    assert len(pred_string_list) == len(target_string_list)

    # if don't remove repetitive generated tokens
    # candidate_string_list = pred_string_list
    # remove repetitive generated tokens
    candidate_string_list = []
    for text in pred_string_list:
        tokens = []
        for token in text.split():
            if len(tokens) == 0 or token != tokens[-1]:
                tokens.append(token)
        candidate_string_list.append(" ".join(tokens))


    epoch_loss = running_loss / dataset_sizes['test']
    # print('test loss: {:4f}'.format(epoch_loss))
    tnsre_scores = my_evaluator.score(predictions=candidate_string_list, references=target_string_list)
    print(tnsre_scores)
    logger.info("\nMode: test ==> "
                "ROUGE_1 {} ROUGE_2 {} ROUGE_L {} WER {} BLEU_1 {} BLEU_2 {} BLEU_3 {} BLEU_4 {}.\n ".format(tnsre_scores["rouge1"],
                                                                                                             tnsre_scores["rouge2"],
                                                                                                             tnsre_scores["rougel"],
                                                                                                             tnsre_scores["wer"],
                                                                                                             tnsre_scores["nltk_bleu_1"],
                                                                                                             tnsre_scores["nltk_bleu_2"],
                                                                                                             tnsre_scores["nltk_bleu_3"],
                                                                                                             tnsre_scores["nltk_bleu_4"]))

    """ calculate corpus bleu score """
    weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
    bleu_scores = 0
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        # print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        logger.info('corpus BLEU-{} score:{}'.format(len(list(weight)), corpus_bleu_score))
        if len(list(weight)) == 4:
            bleu_scores=corpus_bleu_score
    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    # print(rouge_scores)
    logger.info(rouge_scores)
    # rouge_1_r = rouge_scores['rouge-1']['r']
    # rouge_1_p = rouge_scores['rouge-1']['p']
    # rouge_1_f = rouge_scores['rouge-1']['f']
    # rouge_1_total = (rouge_1_r + rouge_1_p + rouge_1_f)*100

    # NOTE:only use bleu_scores
    # all_scores = bleu_scores*100 + rouge_1_total

    # # if best_scores<all_scores:
    # best_scores = all_scores
    # print(best_scores)
    # # logger.info("best scores of belu and rouge 1 :{}\n".format(best_scores))
    # torch.save(model.state_dict(), checkpoint_path_save)
    # print(f'update the checkpoint: {checkpoint_path_save}')
    # logger.info("update the checkpoint in {}".format(checkpoint_path_save))
    # return best_scores
    # model.eval()  # Set model to evaluate mode
    #
    # target_tokens_list = []
    # target_string_list = []
    # pred_tokens_list = []
    # pred_string_list = []
    # with open(output_all_results_path, 'w') as f:
    #     for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders['test']):
    #         # load in batch
    #         input_embeddings_batch = input_embeddings.to(device).float()
    #         input_masks_batch = input_masks.to(device)
    #         target_ids_batch = target_ids.to(device)
    #         input_mask_invert_batch = input_mask_invert.to(device)
    #
    #         target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
    #         target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
    #         # print('target ids tensor:',target_ids_batch[0])
    #         # print('target ids:',target_ids_batch[0].tolist())
    #         # print('target tokens:',target_tokens)
    #         # print('target string:',target_string)
    #         f.write(f'target string: {target_string}\n')
    #
    #         # add to list for later calculate bleu metric
    #         target_tokens_list.append([target_tokens])
    #         target_string_list.append(target_string)
    #
    #         """replace padding ids in target_ids with -100"""
    #         target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
    #
    #         # target_ids_batch_label = target_ids_batch.clone().detach()
    #         # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100
    #
    #         predictions = model.generate(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
    #                                      target_ids_batch,
    #                                      max_length=256,
    #                                      num_beams=1,
    #                                      do_sample=False,
    #                                      # num_beams=5,encoder_no_repeat_ngram_size =1,
    #                                      # do_sample=True, top_k=15,temperature=0.5,num_return_sequences=5,
    #                                      # early_stopping=True
    #                                      )
    #         predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    #         # predicted_string=predicted_string.squeeze()
    #         predicted_string = predicted_string[0]
    #         predicted_string = predicted_string.split('</s></s>')[0].replace('<s>', '')
    #         predictions = tokenizer.encode(predicted_string)
    #         # print('predicted string:',predicted_string)
    #         f.write(f'predicted string: {predicted_string}\n')
    #         f.write(f'################################################\n\n\n')
    #
    #         # convert to int list
    #         # predictions = predictions.tolist()
    #         truncated_prediction = []
    #         for t in predictions:
    #             if t != tokenizer.eos_token_id:
    #                 truncated_prediction.append(t)
    #             else:
    #                 break
    #         pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
    #         # print('predicted tokens:',pred_tokens)
    #         pred_tokens_list.append(pred_tokens)
    #         pred_string_list.append(predicted_string)
    #         # print('################################################')
    #         # print()
    #
    # """ calculate corpus bleu score """
    # weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
    # for weight in weights_list:
    #     # print('weight:',weight)
    #     corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
    #     print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
    #
    # print()
    # """ calculate rouge score """
    # rouge = Rouge()
    # rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    # print(rouge_scores)



def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


def save_require_grad_layers_to_txt(model, filename):
    with open(filename, 'w') as file:
        file.write('require_grad layers:\n')
        for name, param in model.named_parameters():
            if param.requires_grad:
                file.write(f'{name}\n')
    print(f"Saved require_grad layers to '{filename}'.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EEG-Text')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    args = vars(parser.parse_args())
    args = read_configuration(args["config"])

    ''' config param'''

    num_epoch_fintune = args['num_epoch_fintune']

    lr_finetune = args['lr_finetune']

    batch_size = args['batch_size']

    # args['model_name'] = "cet-mae"

    model_name = args['model_name']

    args["log_dir"] = "./logs/logging/cet_mae_eeg2text/"

    init_logger(args)
    logger = getLogger()
    logger.info("dataset_path:{}".format(args["dataset_path"]))
    logger.info("lr_finetune:{}".format(lr_finetune))
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    # task_name = 'task1_task2_taskNRv2_taskTSRv2'
    task_name = args['task_name']

    save_path = args['save_path']

    print(f'[INFO]using model: {model_name}')

    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    # if skip_step_clip:
    #     save_name = f'{model_name}_skipstep1_b{batch_size}_{num_epoch_clip}_{lr_clip}_fintune_{num_epoch_fintune}_{lr_fintune}'
    # else:
    #     save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_clip_{num_epoch_clip}_{lr_clip}_fintune_{num_epoch_fintune}_{lr_fintune}'

    """ task num """
    task_num = args["dataset_path"].split("_")[-1][0]

    """ checkpoint save"""
    # save_name = f'{task_num}_tasks_{model_name}_b{batch_size}_fintune_{num_epoch_fintune}_{lr_finetune}'
    # output_checkpoint_name_best = save_path + f'/best/{save_name}.pt'
    # output_checkpoint_name_last = save_path + f'/last/{save_name}.pt'
    save_name = args["text_rusults"]
    output_all_results_path = f'./result/{save_name}.txt'
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    print()

    """save config"""


    tokenizer = MvpTokenizer.from_pretrained("./models/huggingface/mvp_multi")


    """ dataset """
    train_set = EEG_dataset(path=args["dataset_path"] + "train")
    valid_set = EEG_dataset(path=args["dataset_path"] + "valid")
    test_set = EEG_dataset(path=args["dataset_path"]+ "test")
    dataset_sizes = {'train': len(train_set), 'dev': len(valid_set),'test':len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(valid_set))

    """ dataloader """
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    dataloaders = {'train': train_dataloader, 'dev': valid_dataloader,'test':test_dataloader}

    pretrained = MvpForConditionalGeneration.from_pretrained('./models/huggingface/mvp_multi')
    model = BrainMVP(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                       pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)


    model.to(device)

    eeg2text_checkpoint = args["eeg2text_checkpoint"]
    print(eeg2text_checkpoint)
    logger.info(eeg2text_checkpoint)
    model.load_state_dict(torch.load(eeg2text_checkpoint), strict=False)

    bleu_scores = eval_model(dataloaders, device, tokenizer, model,output_all_results_path=output_all_results_path)