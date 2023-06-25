import os
import sys
from argparse import ArgumentParser

import torch
from pathlib2 import Path
from torch.utils.data import DataLoader

from Utils import utils
from Utils.bert_training_utils import train, evaluate
from transformers import BertTokenizer
from constants import *
from custom_subject_data_loader import Subject_Dataset

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def import_model(model_type, model_name):
    module = __import__('models.' + model_type, fromlist=['models'])
    return module.create(model_name)


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)
    tokenizer = None

    if not args.test:
        try:
            tokenizer_model_type = utils.config['tokenizer_type']
            tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type)
        except Exception as e:
            print(f"No Tokenizer defined in Config file: {e}")

        if isinstance(tokenizer, type(None)):
            tokenizer = BertTokenizer.from_pretrained(DEFAULT_TOKENIZER_TYPE)

    if not args.infer:
        if args.custom_data:
            dataset_path = Path(utils.config['custom_data_path'])
            print(dataset_path)

            train_dataset = Subject_Dataset(dataset_path / 'train', tokenizer=tokenizer)
            dev_dataset = Subject_Dataset(dataset_path / 'dev', tokenizer=tokenizer)
            test_dataset = Subject_Dataset(dataset_path / 'test', tokenizer=tokenizer)
        else:
            # Can be implemented for some other dataset.
            # Custom dataset class can be created according to preprocessig needs.
            train_dataset = ""
            dev_dataset = ""
            test_dataset = ""

        train_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers)
        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, shuffle=False,
                            num_workers=args.num_workers)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False,
                             num_workers=args.num_workers)

    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        try:
            model_type = utils.config['model_type']
            model_name = utils.config['model_name']
        except Exception as E:
            print(f"No model_type or model_name in config file {E}.  Using Defaults.")
            model_type = DEFAULT_MODEL_TYPE
            model_name = DEFAULT_MODEL_NAME

        model = import_model(model_type, model_name)
    elif args.load_from:
        print("Model_Path: ", args.load_from)
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)

    print(f"Here is model {model}")

    if  args.train:
        train(model, train_dl, dev_dl, 1e-6, args.epochs, checkpoint_path)
    else:
        if not args.load_from:
            print("Please provide model_path to load from for evalution or mention --train for training")
            return

        evaluate (model,test_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=2)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=1)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run', action='store_true')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--custom_data', help='Use Custom Dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--infer', help='inference_dir', type=str)

    main(parser.parse_args())
