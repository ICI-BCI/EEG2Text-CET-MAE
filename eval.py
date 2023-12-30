import time
import argparse

from hf_metrics_vanilla import MyEvaluator
from utils import *
from torch.utils.data import DataLoader
from dataset import EEG_dataset
from model import BrainTranslator
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration


def test(test_dataloader, device, model, use_norm_eeg):
    my_evaluator = MyEvaluator()

    inference_time = []
    generated_text = []
    reference_text = []
    file_names = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            input_embeddings, non_normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,file_name = batch
            file_names.append(str(file_name[0]))
            if use_norm_eeg:
                inputs_embeds_batch = input_embeddings.to(device).float()
            else:
                inputs_embeds_batch = non_normalized_input_embeddings.to(device).float()

            inputs_attn_mask_batch = input_attn_mask.to(device)
            inputs_attn_mask_invert_bacth = input_attn_mask_invert.to(device)
            target_input_ids_batch = target_ids.to(device)
            reference = tokenizer.decode(target_input_ids_batch[0], skip_special_tokens=True)

            """replace padding ids in target_ids with -100"""
            target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100

            target_mask_batch = target_mask.to(device)  # maybe duplicated with the above codes
            t0 = time.perf_counter()
            output_dict = model(inputs_embeds=inputs_embeds_batch,
                                attention_mask=inputs_attn_mask_batch,
                                attention_mask_invert=inputs_attn_mask_invert_bacth,
                                decoder_attention_mask=target_mask_batch,  # TODO？
                                labels=target_input_ids_batch,
                                return_dict=True)
            t1 = time.perf_counter()
            final_logits = output_dict["logits"]

            probs = final_logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            generated = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')

            generated_text.append(generated)
            reference_text.append(reference)
            inference_time.append((t1 - t0) * 1000)

        assert len(generated_text) == len(reference_text)

        # remove repetitive generated tokens
        candidate_text = []
        for text in generated_text:
            tokens = []
            for token in text.split():
                if len(tokens) == 0 or token != tokens[-1]:
                    tokens.append(token)
            candidate_text.append(" ".join(tokens))

        saved_file_path = config["output_dir"]
        model_name = config["finetuned_model_dir"].split("/")[-1].replace(".pt", "")

        if not os.path.exists(saved_file_path):
            os.makedirs(saved_file_path)

        saved_file = os.path.join(saved_file_path, f"{model_name}_output_text.txt")
        fout = open(saved_file, "w")
        for i in range(len(candidate_text)):
            fout.write(file_names[i] + "\n")
            fout.write("Generated text: " + candidate_text[i].strip() + "\n")
            fout.write("Reference text: " + reference_text[i].strip() + "\n\n")
        fout.close()

        candidate_text = [text.lower().strip() for text in candidate_text]
        reference_text = [text.lower().strip() for text in reference_text]

        scores = my_evaluator.score(predictions=candidate_text, references=reference_text)

        print(scores)
        avg_time = sum(inference_time) / len(inference_time)
        scores["inference_latency"] = avg_time

        metric_file = os.path.join(saved_file_path, f"{model_name}_metric.txt")
        fout = open(metric_file, "w")
        fout.write(json.dumps(scores) + "\n")
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Script')
    parser.add_argument('-c', '--config', help='path to file', required=True)
    args = vars(parser.parse_args())
    config = read_configuration(args["config"])

    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)
    logging.info(device)

    exp_name = config["exp_name"]

    test_set = EEG_dataset(path=config["data_dir"] + "test")
    logging.info(f'test_set size: {len(test_set)}')

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # """tokenizer and model"""
    # tokenizer = BartTokenizer.from_pretrained(config["pretrained_model"])
    # bart = BartForConditionalGeneration.from_pretrained(config["pretrained_model"])

    """tokenizer and model"""
    if "bart" in config["pretrained_model"]:
        tokenizer = BartTokenizer.from_pretrained(config["pretrained_model"])
        pre_trained_seq2seq = BartForConditionalGeneration.from_pretrained(config["pretrained_model"])
    elif "t5" in config["pretrained_model"]:
        tokenizer = T5Tokenizer.from_pretrained(config["pretrained_model"])
        pre_trained_seq2seq = T5ForConditionalGeneration.from_pretrained(config["pretrained_model"])
    elif "pegasus" in config["pretrained_model"]:
        tokenizer = PegasusTokenizer.from_pretrained(config["pretrained_model"])
        pre_trained_seq2seq = PegasusForConditionalGeneration.from_pretrained(config["pretrained_model"])

    model = BrainTranslator(pre_trained_seq2seq, pre_encoder_in_dim=config["pre_encoder_in_dim"],
                            pre_encoder_out_dim=config["pre_encoder_out_dim"],
                            pre_encoder_head=config["pre_encoder_head"],
                            pre_encoder_ffnn_dim=config["pre_encoder_ffnn_dim"],
                            pre_encoder_layer_num=config["pre_encoder_layer_num"],
                            pre_encoder_name=config["pre_encoder_name"])

    model.load_state_dict(torch.load(config["finetuned_model_dir"]))
    logger.info("Load fine-tuned model from {}.".format(config["finetuned_model_dir"]))
    model.to(device)
    model.eval()

    logging.info('=== start testing ... ===')
    test(test_dataloader, device, model, use_norm_eeg=config["use_norm_eeg"])
