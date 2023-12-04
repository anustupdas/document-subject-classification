from argparse import ArgumentParser
from Utils import utils
from transformers import BertTokenizer
from constants import DEFAULT_TOKENIZER_TYPE
import torch


def predict(test_input, model, label_dict):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    mask = test_input['attention_mask'].to(device)
    input_id = test_input['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask)
    predicted_label = output.argmax(dim=1).tolist()[0]
    class_label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]
    if class_label in ['Random (best rated docs)']:
        print("Accepted")
    else:
        print("Rejected")
    print(class_label)


def main(args):
    utils.read_config_file(args.config)
    label_dict = utils.config['labels_class']
    tokenizer = BertTokenizer.from_pretrained(DEFAULT_TOKENIZER_TYPE)
    text = "It is Very hot in Assam"
    if args.text:
        text = args.text
    else:
        text = input("Enter text:")
    encoded_text = tokenizer(str(text), padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    if not args.load_from:
        print("Please provide model_path to load from for inferencing")
        return

    print("Loading Model from : ", args.load_from)
    with open(args.load_from, 'rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))

    predict(encoded_text, model, label_dict)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--input_directory', help='folder to the text files to be inferred', default=None, type=str)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--input_file', help='path to the text file to be inferred', default=None, type=str)
    parser.add_argument('--text', help='raw text as input', default="Artificial intelligence in computer science are "
                                                                    "achiving new heights", type=str)
    main(parser.parse_args())
