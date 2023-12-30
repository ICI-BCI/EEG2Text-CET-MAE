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
from model import BrainTranslator,BrainMVPos,BrainMultiStreamMVP
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

def train_model(dataloaders, device, model, criterion, optimizer, scheduler,early_stopping, num_epochs=25,
                checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt',
                checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt',
                output_all_results_path='./results/temp.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    best_valid_loss = None
    best_valid_bleu = None
    best_valid_rouge1_r = None
    best_valid_rouge1_p = None
    best_valid_rouge1_f = None
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_rouge1 = 0
    best_bleu_scores = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            elif phase=='dev':
                model.eval()  # Set model to evaluate mode
            # elif phase == 'test':
            #     bleu_scores = eval_model(dataloaders, device, tokenizer, criterion, model,
            #                output_all_results_path=output_all_results_path,checkpoint_path_save=checkpoint_path_best,best_scores=best_bleu_scores)
            #     best_bleu_scores = bleu_scores
            #     continue
            running_loss = 0.0

            # Iterate over data. 对数据进行迭代
            # 从 dataloaders中的inputs中提取
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(
                    dataloaders[phase]):
                # for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders[phase]):
                # load in batch
                # TODO:看一下input_embeddings 的维度
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)

                if phase == 'dev':
                    target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
                    target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
                    # add to list for later calculate bleu metric
                    target_tokens_list.append([target_tokens])
                    target_string_list.append(target_string)

                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                        target_ids_batch)

                """calculate loss"""
                # NOTE: my criterion not used
                # hugging face 中的Bart 里面自己可以计算loss吧？
                loss = seq2seqLMoutput.loss  # use the BART language modeling loss

                # logits = seq2seqLMoutput.logits 大小为 Tensor:(32,58,50267)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    # optimizer.step_and_update_lr()
                    optimizer.step()
                #
                if phase == 'dev':
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
                    assert len(pred_string_list) == len(target_string_list)

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # deep copy the model


            if phase=='dev':
                # calculate corpus bleu score
                weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
                bleu_scores = 0
                for weight in weights_list:
                    # print('weight:',weight)
                    corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
                    # print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
                    # logger.info('{} corpus BLEU-{} score:{}'.format(phase, len(list(weight)), corpus_bleu_score))
                    # bleu_scores += corpus_bleu_score
                # print()
                # calculate rouge score
                rouge = Rouge()
                rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
                # print(rouge_scores)
                # logger.info(rouge_scores)
                rouge_1_r = rouge_scores['rouge-1']['r']
                rouge_1_p = rouge_scores['rouge-1']['p']
                rouge_1_f = rouge_scores['rouge-1']['f']
                # rouge_1_total = rouge_1_r + rouge_1_p + rouge_1_f
                # print("rouge_1_total:", rouge_1_total)
                # logger.info('{} rouge_1_total: {}'.format(phase,rouge_1_total))
            if phase == 'dev':
                save = False
                if best_valid_loss is None or best_valid_loss[1] > epoch_loss:
                    best_valid_loss = (epoch, epoch_loss)
                    save = True

                # BLEU-4
                if best_valid_bleu is None or best_valid_bleu[1] < corpus_bleu_score:
                    best_valid_bleu = (epoch, corpus_bleu_score)
                    save = True

                # ROUGE-1 RECALL
                if best_valid_rouge1_r is None or best_valid_rouge1_r[1] < rouge_1_r:
                    best_valid_rouge1_r = (epoch, rouge_1_r)
                    save = True

                # ROUGE-1 Presion
                if best_valid_rouge1_p is None or best_valid_rouge1_p[1] < rouge_1_p:
                    best_valid_rouge1_p = (epoch, rouge_1_p)
                    save = True

                # ROUGE-1 F1
                if best_valid_rouge1_f is None or best_valid_rouge1_f[1] < rouge_1_f:
                    best_valid_rouge1_f = (epoch, rouge_1_f)
                    save = True


                # if save and epoch >=20:
                if save:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    '''save checkpoint'''
                    checkpoint_path_best_1 = f"{checkpoint_path_best}_epoch{epoch}.pt"  # 添加 epoch 信息
                    torch.save(model.state_dict(), checkpoint_path_best_1)
                    print(f'update best on dev checkpoint: {checkpoint_path_best_1}')
                    logger.info("update best on dev checkpoint:{}".format(checkpoint_path_best_1))

        print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))
    # torch.save(model.state_dict(), checkpoint_path_last)
    # print(f'update last checkpoint: {checkpoint_path_last}')
    print("Finish training, please take a look.")
    print("The best loss {} in epoch {},".format(best_valid_loss[1], best_valid_loss[0]))
    print("The best BLEU-4 {} in epoch {},".format(best_valid_bleu[1], best_valid_bleu[0]))
    print("The best ROUGE1-R {} in epoch {},".format(best_valid_rouge1_r[1], best_valid_rouge1_r[0]))
    print("The best ROUGE1_P {} in epoch {},".format(best_valid_rouge1_p[1], best_valid_rouge1_p[0]))
    print("The best ROUGE1_F1 {} in epoch {}".format(best_valid_rouge1_f[1], best_valid_rouge1_f[0]))
    logger.info("\n"
                "The best loss {} in epoch {},"
                "the best BLEU-4 {} in epoch {}, "
                "the best ROUGE1-R {} in epoch {}, "
                "the best ROUGE1_P {} in epoch {}, "
                "the best ROUGE1_F1 {} in epoch {}. \n".format( best_valid_loss[1], best_valid_loss[0], best_valid_bleu[1], best_valid_bleu[0],
        best_valid_rouge1_r[1], best_valid_rouge1_r[0], best_valid_rouge1_p[1], best_valid_rouge1_p[0], best_valid_rouge1_f[1],best_valid_rouge1_f[0]))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return


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

    # args['model_name'] = "cet-mae_bart"

    model_name = args['model_name']

    # args["log_dir"] = "./logs/logging/cet_mae_eeg2text/"

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
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    save_name = f'{task_num}_tasks_{model_name}_b{batch_size}_cet_mae_fintune_{num_epoch_fintune}_{lr_finetune}'
    output_checkpoint_name_best = save_path + f'/{save_name}'
    output_all_results_path = f'./result/{task_num}--{task_name}-{model_name}-all_decoding_results.txt'
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

    if model_name in ['BrainTranslator', 'BrainTranslatorNaive', 'BrainTranslatorLSTM',"cet-mae_bart"]:
        # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('./models/huggingface/bart-large')
    elif model_name == 'BrainT5':
        tokenizer = T5Tokenizer.from_pretrained("./models/huggingface/t5-large")
    elif model_name == 'BrainMVP':
        tokenizer = MvpTokenizer.from_pretrained("./models/huggingface/mvp")
    elif model_name in ['BrainMVPlus', 'BrainMVPlusPos','cscl_cet_mae_brainmvp',"cet-mae"]:
        tokenizer = MvpTokenizer.from_pretrained("./models/huggingface/mvp_multi")
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True

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

    ''' set up model '''
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained('./models/huggingface/bart-large')
        model = BrainTranslator(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                           pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)


    elif model_name == "cet-mae":
        pretrained = MvpForConditionalGeneration.from_pretrained('./models/huggingface/mvp_multi')
        model = BrainMultiStreamMVP(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                           pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)
        # wo_multistream
        # model = BrainMVPos(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
        #                    pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)

    elif model_name == "cet-mae_bart":
        pretrained = BartForConditionalGeneration.from_pretrained('./models/huggingface/bart-large')
        print("the LLM is bart")
        model = BrainMultiStreamMVP(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                           pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)

    elif model_name == "cet-mae_roberta":
        pretrained = BartForConditionalGeneration.from_pretrained('./models/huggingface/xlm-roberta-large')
        print("the LLM is bart")
        model = BrainMultiStreamMVP(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                           pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)

    elif model_name == 'BertGeneration':
        pretrained = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
        model = BrainTranslator(pretrained, pre_encoder_in_dim = 105 * len(bands_choice),
                           pre_encoder_out_dim = 1024,pre_encoder_head = 8,pre_encoder_ffnn_dim = 2048)


    model.to(device)

    ''' training loop '''
    ######################################################
    '''load Contrastive EEG-Text Masked Autoencoder(CET-MAE) checkpoint'''
    ######################################################
    cet_mae_checkpoint = args["cet_mae_checkpoint"]
    print(cet_mae_checkpoint)
    logger.info(cet_mae_checkpoint)
    model.load_state_dict(torch.load(cet_mae_checkpoint), strict=False)
    print("success load CET-MAE checkpoints")
    logger.info("success load CET-MAE checkpoints")



    # 论文选用的是 跳过第一步，从第二步开始
    ######################################################
    '''step one trainig: update whole model for a few iterations'''
    ######################################################

    # 'fc_eeg.weight',
    # 'fc_eeg.bias',
    # 是否冻结权 重
    is_frozen = False
    print("is_froze:",is_frozen)
    logger.info("is_frozen: {}".format(is_frozen))
    if is_frozen:
        print("frozen EEG encoder, fine-tuning fc & LLM")
        # desired_weights = [
        #     'eeg_encoder_layer.self_attn.in_proj_weight',
        #     'eeg_encoder_layer.self_attn.in_proj_bias',
        #     'eeg_encoder_layer.self_attn.out_proj.weight',
        #     'eeg_encoder_layer.self_attn.out_proj.bias',
        #     'eeg_encoder_layer.linear1.weight',
        #     'eeg_encoder_layer.linear1.bias',
        #     'eeg_encoder_layer.linear2.weight',
        #     'eeg_encoder_layer.linear2.bias',
        #     'unify_encoder_layer.self_attn.in_proj_weight',
        #     'unify_encoder_layer.self_attn.in_proj_bias',
        #     'unify_encoder_layer.self_attn.out_proj.weight',
        #     'unify_encoder_layer.self_attn.out_proj.bias',
        #     'unify_encoder_layer.linear1.weight',
        #     'unify_encoder_layer.linear1.bias',
        #     'unify_encoder_layer.linear1_e.weight',
        #     'unify_encoder_layer.linear1_e.bias',
        #     'unify_encoder_layer.linear1_t.weight',
        #     'unify_encoder_layer.linear1_t.bias',
        #     'unify_encoder_layer.linear2.weight',
        #     'unify_encoder_layer.linear2.bias',
        #     'unify_encoder_layer.linear2_e.weight',
        #     'unify_encoder_layer.linear2_e.bias',
        #     'unify_encoder_layer.linear2_t.weight',
        #     'unify_encoder_layer.linear2_t.bias',
        #     'unify_branch.layers.0.self_attn.in_proj_weight',
        #     'unify_branch.layers.0.self_attn.in_proj_bias',
        #     'unify_branch.layers.0.self_attn.out_proj.weight',
        #     'unify_branch.layers.0.self_attn.out_proj.bias',
        #     'unify_branch.layers.0.linear1.weight',
        #     'unify_branch.layers.0.linear1.bias',
        #     'unify_branch.layers.0.linear1_e.weight',
        #     'unify_branch.layers.0.linear1_e.bias',
        #     'unify_branch.layers.0.linear1_t.weight',
        #     'unify_branch.layers.0.linear1_t.bias',
        #     'unify_branch.layers.0.linear2.weight',
        #     'unify_branch.layers.0.linear2.bias',
        #     'unify_branch.layers.0.linear2_e.weight',
        #     'unify_branch.layers.0.linear2_e.bias',
        #     'unify_branch.layers.0.linear2_t.weight',
        #     'unify_branch.layers.0.linear2_t.bias',
        #     'e_branch.layers.0.self_attn.in_proj_weight',
        #     'e_branch.layers.0.self_attn.in_proj_bias',
        #     'e_branch.layers.0.self_attn.out_proj.weight',
        #     'e_branch.layers.0.self_attn.out_proj.bias',
        #     'e_branch.layers.0.linear1.weight',
        #     'e_branch.layers.0.linear1.bias',
        #     'e_branch.layers.0.linear2.weight',
        #     'e_branch.layers.0.linear2.bias',
        #     'e_branch.layers.1.self_attn.in_proj_weight',
        #     'e_branch.layers.1.self_attn.in_proj_bias',
        #     'e_branch.layers.1.self_attn.out_proj.weight',
        #     'e_branch.layers.1.self_attn.out_proj.bias',
        #     'e_branch.layers.1.linear1.weight',
        #     'e_branch.layers.1.linear1.bias',
        #     'e_branch.layers.1.linear2.weight',
        #     'e_branch.layers.1.linear2.bias',
        #     'e_branch.layers.2.self_attn.in_proj_weight',
        #     'e_branch.layers.2.self_attn.in_proj_bias',
        #     'e_branch.layers.2.self_attn.out_proj.weight',
        #     'e_branch.layers.2.self_attn.out_proj.bias',
        #     'e_branch.layers.2.linear1.weight',
        #     'e_branch.layers.2.linear1.bias',
        #     'e_branch.layers.2.linear2.weight',
        #     'e_branch.layers.2.linear2.bias',
        #     'e_branch.layers.3.self_attn.in_proj_weight',
        #     'e_branch.layers.3.self_attn.in_proj_bias',
        #     'e_branch.layers.3.self_attn.out_proj.weight',
        #     'e_branch.layers.3.self_attn.out_proj.bias',
        #     'e_branch.layers.3.linear1.weight',
        #     'e_branch.layers.3.linear1.bias',
        #     'e_branch.layers.3.linear2.weight',
        #     'e_branch.layers.3.linear2.bias',
        #     'e_branch.layers.4.self_attn.in_proj_weight',
        #     'e_branch.layers.4.self_attn.in_proj_bias',
        #     'e_branch.layers.4.self_attn.out_proj.weight',
        #     'e_branch.layers.4.self_attn.out_proj.bias',
        #     'e_branch.layers.4.linear1.weight',
        #     'e_branch.layers.4.linear1.bias',
        #     'e_branch.layers.4.linear2.weight',
        #     'e_branch.layers.4.linear2.bias',
        #     'e_branch.layers.5.self_attn.in_proj_weight',
        #     'e_branch.layers.5.self_attn.in_proj_bias',
        #     'e_branch.layers.5.self_attn.out_proj.weight',
        #     'e_branch.layers.5.self_attn.out_proj.bias',
        #     'e_branch.layers.5.linear1.weight',
        #     'e_branch.layers.5.linear1.bias',
        #     'e_branch.layers.5.linear2.weight',
        #     'e_branch.layers.5.linear2.bias',
        #     'fc_eeg.weight',
        #     'fc_eeg.bias'
        # ]
        desired_weights = [
            'eeg_encoder_layer.self_attn.in_proj_weight',
            'eeg_encoder_layer.self_attn.in_proj_bias',
            'eeg_encoder_layer.self_attn.out_proj.weight',
            'eeg_encoder_layer.self_attn.out_proj.bias',
            'unify_encoder_layer.self_attn.in_proj_weight',
            'unify_encoder_layer.self_attn.in_proj_bias',
            'unify_encoder_layer.self_attn.out_proj.weight',
            'unify_encoder_layer.self_attn.out_proj.bias',
            'unify_branch.layers.0.self_attn.in_proj_weight',
            'unify_branch.layers.0.self_attn.in_proj_bias',
            'unify_branch.layers.0.self_attn.out_proj.weight',
            'unify_branch.layers.0.self_attn.out_proj.bias',
            'e_branch.layers.0.self_attn.in_proj_weight',
            'e_branch.layers.0.self_attn.in_proj_bias',
            'e_branch.layers.0.self_attn.out_proj.weight',
            'e_branch.layers.0.self_attn.out_proj.bias',
            'e_branch.layers.1.self_attn.in_proj_weight',
            'e_branch.layers.1.self_attn.in_proj_bias',
            'e_branch.layers.1.self_attn.out_proj.weight',
            'e_branch.layers.1.self_attn.out_proj.bias',
            'e_branch.layers.2.self_attn.in_proj_weight',
            'e_branch.layers.2.self_attn.in_proj_bias',
            'e_branch.layers.2.self_attn.out_proj.weight',
            'e_branch.layers.2.self_attn.out_proj.bias',
            'e_branch.layers.3.self_attn.in_proj_weight',
            'e_branch.layers.3.self_attn.in_proj_bias',
            'e_branch.layers.3.self_attn.out_proj.weight',
            'e_branch.layers.3.self_attn.out_proj.bias',
            'e_branch.layers.4.self_attn.in_proj_weight',
            'e_branch.layers.4.self_attn.in_proj_bias',
            'e_branch.layers.4.self_attn.out_proj.weight',
            'e_branch.layers.4.self_attn.out_proj.bias',
            'e_branch.layers.5.self_attn.in_proj_weight',
            'e_branch.layers.5.self_attn.in_proj_bias',
            'e_branch.layers.5.self_attn.out_proj.weight',
            'e_branch.layers.5.self_attn.out_proj.bias'
        ]

        for name, param in model.named_parameters():
            if name in desired_weights:
                param.requires_grad = False
            else:
                param.requires_grad = True

    ''' set up optimizer and scheduler'''
    warm_steps = 10000
    # optimizer_step3 = ScheduledOptim(
    #     optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()),
    #                betas=(0.9, 0.98), eps=1e-4, lr=lr_finetune, weight_decay=1e-2), 1024, warm_steps)
        # optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
        #             betas=(0.9, 0.98), eps=1e-9, lr=step2_lr, weight_decay=1e-2), 1024, step2_lr,warm_steps)
    early_stopping = EarlyStopper(patience=4, min_delta=0.01)

    # 第二种训练方式
    optimizer = build_optimizer(args, model, mode="finetune")
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(parameters, args["lr_finetune"], weight_decay=5e-7, betas=(0.95, 0.999))
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 Fintune Training ... ===')
    # print training layers
    # save_require_grad_layers_to_txt(model,'./save_require_grad_layers.txt')
    # show_require_grad_layers(model)

    '''main loop'''
    # trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
    # exp_lr_scheduler_step1是随便填的因为也用不到
    train_model(dataloaders, device, model, criterion, optimizer, scheduler, early_stopping,
                num_epochs=num_epoch_fintune, checkpoint_path_best=output_checkpoint_name_best,
                output_all_results_path=output_all_results_path)
    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))