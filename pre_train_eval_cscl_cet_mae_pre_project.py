import os
import argparse
from utils import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, \
    BartForSequenceClassification, BertTokenizer, BertConfig, \
    T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config, \
    MvpTokenizer, MvpConfig, MvpForConditionalGeneration,MvpModel
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset
# from model_mae import CETMAE_project_late
from model_mae import CETMAE_project_pre
from optim_new import *
from contrastive_eeg_pretraining.pre_encoder import cscl_model_cet_mae
import math

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


def train_mae(train_dataloader, valid_dataloader, model, model_cscl,optimizer, scheduler,tokenizer,
               early_stopping,num_epochs, checkpoint_path,checkpoint_name,checkpoint_eeg_encoder):

    best_loss = 100000000000
    best_mae_loss = 100000000000
    best_mlm_loss = 100000000000
    best_c_loss = 100000000000
    best_sim_loss = 100000000000
    # best_model_wts = copy.deepcopy(model.state_dict())
    # checkpoint_eeg_encoder= "model_cet_mae_eeg_encoder_mask_25.pt"
    for epoch_idx in range(0, num_epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        train_mlm = 0
        train_c = 0
        train_sim = 0
        t0 = time.time()
        logger.info("Epoch {}".format(epoch_idx))
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # target_ids, target_mask,mvp_target_tokenized,text 是纯文本信息
            input_embeddings, non_normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,mvp_target_tokenized,text = batch

            """
            ############################# text ################################
            """
            mvp_target_tokenized = mvp_target_tokenized.to(device)
            mvp_target_tokenized['input_ids'] = mvp_target_tokenized['input_ids'].squeeze()
            mvp_target_tokenized['attention_mask'] = mvp_target_tokenized['attention_mask'].squeeze()
            # 创建一个新的列表来存储反转后的注意力掩码
            attention_mask = mvp_target_tokenized['attention_mask']
            attention_mask_invert = attention_mask ^ 1
            mvp_target_tokenized['attention_mask_invert'] = attention_mask_invert
            ########################## EEG #######################################
            inputs_embeds_batch = input_embeddings.to(device).float()
            # inputs_attn_mask_batch = {Tensor:(32,58)} ,仅由 0 1 组成
            inputs_attn_mask_batch = input_attn_mask.to(device)
            # inputs_attn_mask_invert_bacth= {Tensor:(32,58)}
            inputs_attn_mask_invert_bacth = input_attn_mask_invert.to(device)
            target_input_ids_batch = target_ids.to(device)
            """replace padding ids in target_ids with -100"""
            target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100

            """
            ########################### load CSCL ###############################
            """
            inputs_cscl_embeds_batch = model_cscl(inputs_embeds_batch,input_attn_mask_invert_batch=inputs_attn_mask_invert_bacth)


            loss_mae, loss_mlm, loss_c, loss_sim, all_loss = model(inputs_cscl_embeds_batch,inputs_attn_mask_batch,inputs_attn_mask_invert_bacth,
                                                                   mvp_target_tokenized, mask_ratio_e=0.5,mlm_probability=0.5)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            # TODO lr_scheduler
            scheduler.step()


            train_loss += all_loss.item()
            train_mae +=loss_mae.item()
            train_mlm += loss_mlm.item()
            train_c += loss_c.item()
            train_sim += loss_sim.item()

        train_epoch_loss = train_loss/len(train_dataloader)  # TODO make sure of this
        train_epoch_mae_loss = train_mae/len(train_dataloader)
        train_epoch_mlm_loss = train_mlm / len(train_dataloader)
        train_epoch_c_loss = train_c / len(train_dataloader)
        train_epoch_sim_loss = train_sim / len(train_dataloader)
        print('Epoch {} Train Loss: {:.4f}'.format(epoch_idx, train_epoch_loss))
        logger.info("Epoch {} Train Loss: {:.4f}".format(epoch_idx, train_epoch_loss))
        all_train_losses.append(train_epoch_loss)
        all_train_mae_losses.append(train_epoch_mae_loss)
        all_train_mlm_losses.append(train_epoch_mlm_loss)
        all_train_c_losses.append(train_epoch_c_loss)
        all_train_sim_losses.append(train_epoch_sim_loss)

        # training_time = format_time(time.time() - t0)

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_mae = 0
            valid_mlm = 0
            valid_c = 0
            valid_sim = 0
            for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
                input_embeddings, non_normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, mvp_target_tokenized, text = batch

                """
                ############################# text ################################
                """
                mvp_target_tokenized = mvp_target_tokenized.to(device)
                mvp_target_tokenized['input_ids'] = mvp_target_tokenized['input_ids'].squeeze()
                mvp_target_tokenized['attention_mask'] = mvp_target_tokenized['attention_mask'].squeeze()
                attention_mask = mvp_target_tokenized['attention_mask']
                attention_mask_invert = attention_mask ^ 1
                mvp_target_tokenized['attention_mask_invert'] = attention_mask_invert

                ########################## EEG #######################################
                inputs_embeds_batch = input_embeddings.to(device).float()
                # inputs_attn_mask_batch = {Tensor:(32,58)} ,仅由 0 1 组成
                inputs_attn_mask_batch = input_attn_mask.to(device)
                # inputs_attn_mask_invert_bacth= {Tensor:(32,58)}
                inputs_attn_mask_invert_bacth = input_attn_mask_invert.to(device)
                target_input_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100

                """
                ########################### load CSCL ###############################
                """
                inputs_cscl_embeds_batch = model_cscl(inputs_embeds_batch,input_attn_mask_invert_batch=inputs_attn_mask_invert_bacth)

                loss_mae, loss_mlm, loss_c, loss_sim, all_loss = model(inputs_cscl_embeds_batch, inputs_attn_mask_batch, inputs_attn_mask_invert_bacth,
                                                                       mvp_target_tokenized, mask_ratio_e=0.5, mlm_probability=0.5)

                valid_loss += all_loss.item()
                valid_mae += loss_mae.item()
                valid_mlm += loss_mlm.item()
                valid_c += loss_c.item()
                valid_sim += loss_sim.item()
            valid_epoch_loss = valid_loss/len(valid_dataloader)
            valid_epoch_mae_loss = valid_mae/len(valid_dataloader)
            valid_epoch_mlm_loss = valid_mlm / len(valid_dataloader)
            valid_epoch_c_loss = valid_c / len(valid_dataloader)
            valid_epoch_sim_loss = valid_sim / len(valid_dataloader)

            print('Epoch {} Valid Loss: {:.4f}'.format(epoch_idx, valid_epoch_loss))
            logger.info("Epoch {} Valid Loss: {:.4f}".format(epoch_idx, valid_epoch_loss))
            all_valid_losses.append(valid_epoch_loss)
            all_valid_mae_losses.append(valid_epoch_mae_loss)
            all_valid_mlm_losses.append(valid_epoch_mlm_loss)
            all_valid_c_losses.append(valid_epoch_c_loss)
            all_valid_sim_losses.append(valid_epoch_sim_loss)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            best_epoch = epoch_idx
            saved_name = os.path.join(checkpoint_path,checkpoint_name)
            torch.save(model.state_dict(), saved_name)

            print("save the best cet-mae checkpoint")

            encoder_saved_name = os.path.join(checkpoint_path, checkpoint_eeg_encoder)
            desired_state_dict = {key: value for key, value in model.state_dict().items() if key in desired_weights}
            # 保存特定权重
            torch.save(desired_state_dict, encoder_saved_name)

            # best_model_wts = copy.deepcopy(model.state_dict())
        if valid_epoch_mae_loss< best_mae_loss:
            best_mae_loss = valid_epoch_mae_loss
            best_mae_epoch = epoch_idx

        if valid_epoch_mlm_loss< best_mlm_loss :
            best_mlm_loss = valid_epoch_mlm_loss
            best_mlm_epoch = epoch_idx

        if valid_epoch_c_loss< best_c_loss:
            best_c_loss = valid_epoch_c_loss
            best_c_epoch = epoch_idx

        if valid_epoch_sim_loss < best_sim_loss:
            best_sim_loss = valid_epoch_sim_loss
            best_sim_epoch = epoch_idx

        if early_stopping.early_stop(valid_epoch_loss):
            print("We are at epoch:", epoch_idx)
            break

    print("best_epoch:",best_epoch)
    print("best_mae_epoch:",best_mae_epoch)
    print("best_mlm_epoch:",best_mlm_epoch)
    print("best_contrastive_epoch:",best_c_epoch)
    print("best_sim_epoch:",best_sim_epoch)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


def plot_loss_trend(train_losses, valid_losses, save_path, save_title):
    # Plotting the training and validation loss
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='*')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(save_title)
    plt.legend()
    plt.xticks(epochs[::5])  # 设置横坐标刻度为整数

    # 在每个数据点上标注数值或坐标
    # for i, loss in enumerate(train_losses):
    #     plt.text(epochs[i], loss, f'{loss:.4f}', ha='right', va='bottom',fontsize=6)  # 在训练损失数据点上标注数值
    #
    # for i, loss in enumerate(valid_losses):
    #     plt.text(epochs[i], loss, f'{loss:.4f}', ha='right', va='bottom',fontsize=6)  # 在验证损失数据点上标注数值

    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to release memory


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EEG-Text')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    args = vars(parser.parse_args())
    args = read_configuration(args["config"])

    ''' config param'''
    num_epoch_mae = args['num_epoch_pretrain']

    # num_epoch_fintune = args['num_epoch_fintune']
    # lr_clip = args['lr_clip']
    # lr_fintune = args['lr_fintune']

    batch_size = args['batch_size']

    model_name = args['model_name']

    init_logger(args)
    logger = getLogger()

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

    desired_weights = [
        'fc_eeg.weight',
        'fc_eeg.bias',
        'pos_embed_e.segment_coefficients',
        'eeg_encoder_layer.self_attn.in_proj_weight',
        'eeg_encoder_layer.self_attn.in_proj_bias',
        'eeg_encoder_layer.self_attn.out_proj.weight',
        'eeg_encoder_layer.self_attn.out_proj.bias',
        'eeg_encoder_layer.linear1.weight',
        'eeg_encoder_layer.linear1.bias',
        'eeg_encoder_layer.linear2.weight',
        'eeg_encoder_layer.linear2.bias',
        'eeg_encoder_layer.norm1.weight',
        'eeg_encoder_layer.norm1.bias',
        'eeg_encoder_layer.norm2.weight',
        'eeg_encoder_layer.norm2.bias',
        'e_branch.layers.0.self_attn.in_proj_weight',
        'e_branch.layers.0.self_attn.in_proj_bias',
        'e_branch.layers.0.self_attn.out_proj.weight',
        'e_branch.layers.0.self_attn.out_proj.bias',
        'e_branch.layers.0.linear1.weight',
        'e_branch.layers.0.linear1.bias',
        'e_branch.layers.0.linear2.weight',
        'e_branch.layers.0.linear2.bias',
        'e_branch.layers.0.norm1.weight',
        'e_branch.layers.0.norm1.bias',
        'e_branch.layers.0.norm2.weight',
        'e_branch.layers.0.norm2.bias',
        'e_branch.layers.1.self_attn.in_proj_weight',
        'e_branch.layers.1.self_attn.in_proj_bias',
        'e_branch.layers.1.self_attn.out_proj.weight',
        'e_branch.layers.1.self_attn.out_proj.bias',
        'e_branch.layers.1.linear1.weight',
        'e_branch.layers.1.linear1.bias',
        'e_branch.layers.1.linear2.weight',
        'e_branch.layers.1.linear2.bias',
        'e_branch.layers.1.norm1.weight',
        'e_branch.layers.1.norm1.bias',
        'e_branch.layers.1.norm2.weight',
        'e_branch.layers.1.norm2.bias',
        'e_branch.layers.2.self_attn.in_proj_weight',
        'e_branch.layers.2.self_attn.in_proj_bias',
        'e_branch.layers.2.self_attn.out_proj.weight',
        'e_branch.layers.2.self_attn.out_proj.bias',
        'e_branch.layers.2.linear1.weight',
        'e_branch.layers.2.linear1.bias',
        'e_branch.layers.2.linear2.weight',
        'e_branch.layers.2.linear2.bias',
        'e_branch.layers.2.norm1.weight',
        'e_branch.layers.2.norm1.bias',
        'e_branch.layers.2.norm2.weight',
        'e_branch.layers.2.norm2.bias',
        'e_branch.layers.3.self_attn.in_proj_weight',
        'e_branch.layers.3.self_attn.in_proj_bias',
        'e_branch.layers.3.self_attn.out_proj.weight',
        'e_branch.layers.3.self_attn.out_proj.bias',
        'e_branch.layers.3.linear1.weight',
        'e_branch.layers.3.linear1.bias',
        'e_branch.layers.3.linear2.weight',
        'e_branch.layers.3.linear2.bias',
        'e_branch.layers.3.norm1.weight',
        'e_branch.layers.3.norm1.bias',
        'e_branch.layers.3.norm2.weight',
        'e_branch.layers.3.norm2.bias',
        'e_branch.layers.4.self_attn.in_proj_weight',
        'e_branch.layers.4.self_attn.in_proj_bias',
        'e_branch.layers.4.self_attn.out_proj.weight',
        'e_branch.layers.4.self_attn.out_proj.bias',
        'e_branch.layers.4.linear1.weight',
        'e_branch.layers.4.linear1.bias',
        'e_branch.layers.4.linear2.weight',
        'e_branch.layers.4.linear2.bias',
        'e_branch.layers.4.norm1.weight',
        'e_branch.layers.4.norm1.bias',
        'e_branch.layers.4.norm2.weight',
        'e_branch.layers.4.norm2.bias',
        'e_branch.layers.5.self_attn.in_proj_weight',
        'e_branch.layers.5.self_attn.in_proj_bias',
        'e_branch.layers.5.self_attn.out_proj.weight',
        'e_branch.layers.5.self_attn.out_proj.bias',
        'e_branch.layers.5.linear1.weight',
        'e_branch.layers.5.linear1.bias',
        'e_branch.layers.5.linear2.weight',
        'e_branch.layers.5.linear2.bias',
        'e_branch.layers.5.norm1.weight',
        'e_branch.layers.5.norm1.bias',
        'e_branch.layers.5.norm2.weight',
        'e_branch.layers.5.norm2.bias',
    ]



    """save config"""

    if model_name in ['BrainTranslator', 'BrainTranslatorNaive', 'BrainTranslatorLSTM']:
        # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('./models/huggingface/bart-large')
    elif model_name in ['CET-MAE','BrainMVPlus', 'BrainMVPlusPos']:
        tokenizer = MvpTokenizer.from_pretrained("./models/huggingface/mvp_multi")


    """load cscl model & checkpoint"""
    cscl_checkpoint = args["cscl_checkpoint"]
    logger.info("Loading CSCL checkpoint:{}, success".format(cscl_checkpoint))
    model_cscl = cscl_model_cet_mae(pre_encoder_in_dim=840, pre_encoder_head=8, pre_encoder_ffnn_dim=2048)
    model_cscl = model_cscl.to(device)
    model_cscl.load_state_dict(torch.load(cscl_checkpoint), strict=False)
    print("CSCL checkpoint is:",cscl_checkpoint)
    print("Loading CSCL checkpoint success")



    """ dataset """
    train_set = EEG_dataset(path=args["dataset_path"] + "train")
    valid_set = EEG_dataset(path=args["dataset_path"] + "valid")
    test_set = EEG_dataset(path=args["dataset_path"]+ "test")

    train_set = ConcatDataset([train_set,valid_set])
    # dataset_sizes = {'train': len(train_set), 'dev': len(valid_set),'test':len(test_set)}
    dataset_sizes = {'train': len(train_set), 'test': len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]test_set size: ', len(test_set))

    """ dataloader """
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    # valid_dataloader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    # dataloaders = {'train': train_dataloader, 'dev': valid_dataloader,'test':test_dataloader}
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}


    # model = CETMAE_project_late(multi_heads=8,feedforward_dim=2048,trans_layers=6, device=dev)
    model = CETMAE_project_pre(multi_heads=16, feedforward_dim=4096, trans_layers=8, device=dev)
    model.to(device)



    optimizer = build_optimizer(args, model, mode="cet-mae")
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    early_stopping = EarlyStopper(patience=4, min_delta=0.15)
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)
    all_train_losses = []
    all_train_mae_losses = []
    all_train_mlm_losses = []
    all_train_c_losses = []
    all_train_sim_losses = []

    all_valid_losses = []
    all_valid_mae_losses = []
    all_valid_mlm_losses = []
    all_valid_c_losses = []
    all_valid_sim_losses = []

    logger.info(args['cet_mae_lr'])

    print("save_path: ", args["save_path"])
    print("cet_mae_checkpoint_name: ", args["cet_mae_checkpoint_name"])
    logger.info(args["cet_mae_checkpoint_name"])
    print("checkpoint_eeg_encoder: ", args['checkpoint_eeg_encoder'])
    logger.info(args['checkpoint_eeg_encoder'])

    train_mae(train_dataloader, test_dataloader, model, model_cscl,optimizer, scheduler,tokenizer,
              early_stopping,
              num_epochs=num_epoch_mae,
              checkpoint_path=args["save_path"],
              checkpoint_name = args["cet_mae_checkpoint_name"],
              checkpoint_eeg_encoder = args['checkpoint_eeg_encoder'])

    folder_name = args['folder_name']
    directory = f'./loss_plot/{folder_name}'

    # 检查目录是否存在，如果不存在则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_loss_trend(all_train_losses, all_valid_losses, f'{directory}/cet_mae_pretrain_total_loss_project_late.png','CET-MAE Total Loss')
    plot_loss_trend(all_train_mae_losses, all_valid_mae_losses, f'{directory}/cet_mae_pretrain_mae_loss_project_late.png','CET-MAE EEG-MAE Loss')
    plot_loss_trend(all_train_mlm_losses, all_valid_mlm_losses, f'{directory}/cet_mae_pretrain_mlm_loss_project_late.png','CET-MAE Text-MLM Loss')
    plot_loss_trend(all_train_c_losses, all_valid_c_losses, f'{directory}/cet_mae_pretrain_contrastive_loss_project_late.png','CET-MAE EEG-Text Contrastive Loss')
    plot_loss_trend(all_train_sim_losses, all_valid_sim_losses,f'{directory}/cet_mae_pretrain_similarity_loss_project_late.png','CET-MAE EEG Similarity Loss')