import random
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from pathlib2 import Path
import numpy as np

from Utils.data_preparation_utils import *

train_flter_path = "/home/anustup/Desktop/Projects/Personal/studocu_spam_poc/data/clean_train_ids.npy"
test_flter_path = "/home/anustup/Desktop/Projects/Personal/studocu_spam_poc/data/clean_test_ids.npy"
val_flter_path = "/home/anustup/Desktop/Projects/Personal/studocu_spam_poc/data/clean_val_ids.npy"

train_loaded_data = np.load(train_flter_path)
test_loaded_data = np.load(test_flter_path)
val_loaded_data = np.load(val_flter_path)


big_data = []

def split_data_for_training(dataset_for_training):
    unwanted_files = [".DS_Store"]

    # Split DataFrame based on 'is_train' column
    grouped_df = dataset_for_training.groupby('is_train')

    # Separate DataFrames based on True and False values in 'is_train'
    train_df = grouped_df.get_group(True)
    left_df = grouped_df.get_group(False)

    grouped_left_df = left_df.groupby('is_holdout')
    holdout_df = grouped_left_df.get_group(True)
    test_df = grouped_left_df.get_group(False)

    train = []
    val = []
    test = []

    for index, row in train_df.iterrows():
        #if row['document_id'] in train_loaded_data:
            train.append([row['document_id'],row['outcome'], row['group'], row['text']])

    for index, row in test_df.iterrows():
        test.append([row['document_id'],row['outcome'], row['group'], row['text']])

    for index, row in holdout_df.iterrows():
        val.append([row['document_id'],row['outcome'], row['group'], row['text']])


    total_count_data = ("Total", len(dataset_for_training), len(train), len(test), len(val))
    dataset_split_info = (train, val, test)
    print(total_count_data)
    return dataset_split_info



def creat_dataset_for_training(dataset_for_training,out_root_path, max_words = 510):
    subject_data_final = []
    k_list = [2,2]
    for data_partition in dataset_for_training:
        subject_data = []
        for data in tqdm(data_partition, desc ="Processing data"):

            word_count = count_words(data[3])
            text_content = data[3]
            content = [preprocess_content_text(text_content)]


            #checks if no of tokens exceed max token count then breaks the content into x parts
            #having max token count each
            if word_count > max_words:
                big_data.append((data_partition,data[0],word_count))

            if len(k_list) == 2:
                subject_data_content = (data[0], data[3], data [1],data[2])
                subject_data.append(subject_data_content)
            else:

                for part_no, part in enumerate(content):
                    # if len(part) < 5:
                    #     print(subject_file[0], subject_file, part_no, part)

                    part_file_name = str(data[0]) + f"_{part_no}"
                    subject_data_content = (part_file_name, part, data [1],data[2])
                    subject_data.append(subject_data_content)
        print(len(big_data))
        subject_data_final.append(subject_data)



    try:
        save_datasets(subject_data_final, out_root_path)
        print(f"Training data have been saved in {out_root_path}")
    except Exception:
        print(f"Failed to save data due to {Exception}")

def main(args):
    data_root_path = Path(args.data_path)
    out_root_path = Path(args.output_data_path)

    chunks = []
    dataset_for_training = pd.read_csv(data_root_path, chunksize=10000)

    for chunk in dataset_for_training:
        # Perform your operations on the chunk
        chunks.append(chunk)

    # Concatenate all chunks into a single DataFrame
    dataset_for_training = pd.concat(chunks, ignore_index=True)

    split_data = split_data_for_training(dataset_for_training)
    #Have not implemented max token count from argument yet.
    creat_dataset_for_training(split_data,out_root_path)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--max_token_count', help='Maximum number of words per document', type=int, default=512)
    parser.add_argument('--train_split', help='what percentage of total data to be used for training', type=float, default=.7)
    parser.add_argument('--output_data_path', help='path to the data folder', type=str)
    parser.add_argument('--data_path', help='path to the data folder', type=str)

    main(parser.parse_args())
