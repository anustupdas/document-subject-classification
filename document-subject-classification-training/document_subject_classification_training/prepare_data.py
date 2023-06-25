import random
from argparse import ArgumentParser

from tqdm import tqdm
from pathlib2 import Path

from Utils.data_preparation_utils import *


def split_data_for_training(data_folder_path, subject_folders, train_split=.7):
    unwanted_files = [".DS_Store"]

    train = []
    val = []
    test = []
    train_test_split_metadata = []
    total_count, train_count, val_count, test_count = 0, 0, 0, 0

    for subject in subject_folders:

        if subject in unwanted_files:
            continue

        subject_path = os.path.join(data_folder_path, subject)
        subject_files = os.listdir(subject_path)

        train_files = random.sample(subject_files, int(len(subject_files) * train_split))
        remaining_files = [file for file in subject_files if file not in train_files]
        val_files = random.sample(remaining_files, int(len(remaining_files) * .65))
        test_files = [file for file in remaining_files if file not in val_files]

        for t_f in train_files:
            train.append([subject, t_f])

        for v_f in val_files:
            val.append([subject, v_f])

        for ts_f in test_files:
            test.append([subject, ts_f])

        data_split_metadata = (subject, len(subject_files), len(train_files), len(val_files), len(test_files))
        train_test_split_metadata.append(data_split_metadata)

        total_count += len(subject_files)
        train_count += len(train_files)
        val_count += len(val_files)
        test_count += len(test_files)

    total_count_data = ("Total", total_count, train_count, val_count, test_count)
    train_test_split_metadata.append(total_count_data)
    meta_data_df = pd.DataFrame(train_test_split_metadata, columns=['Category', 'Total', 'Train', 'Val', 'Test'])
    dataset_split_info = (train, val, test)
    return dataset_split_info, meta_data_df



def creat_dataset_for_training(dataset_for_training,data_folder_path, df_metadata,out_path, max_words = 510):
    subject_data_final = []

    for data_partition in dataset_for_training:
        subject_data = []
        for subject_file in tqdm(data_partition, desc ="Processing data"):
            # if not (subject_file[0] == "Music" and subject_file[1] == "44245641.txt"):
            #     continue
            subject_path = os.path.join(data_folder_path, subject_file[0])
            subject_file_path = os.path.join(subject_path, subject_file[1])
            word_count = get_word_count(subject_file[1],df_metadata)
            summary = get_summary(subject_file[1],df_metadata)

            #Read the content file
            with open(subject_file_path) as f:
                content = f.readlines()
            text_content = ' '.join([str(word) for word in content])

            content = [preprocess_content_text(text_content)]


            if word_count == 0:
                word_count = count_words(text_content)

            #checks if no of tokens exceed max token count then breaks the content into x parts
            #having max token count each
            if word_count > max_words:
                content = break_text_in_parts(content)

            if len(content) == 1:
                subject_data_content = (subject_file[1], content[0], summary, subject_file[0])
                subject_data.append(subject_data_content)
            else:

                for part_no, part in enumerate(content):
                    if len(part) < 5:
                        print(subject_file[0], subject_file, part_no, part)

                    part_file_name = subject_file[1][:-4] + f"_{part_no}.txt"
                    subject_data_content = (part_file_name, part, summary, subject_file[0])
                    subject_data.append(subject_data_content)
        subject_data_final.append(subject_data)



    try:
        save_datasets(subject_data_final, out_path)
        print(f"Training data have been saved in {out_path}")
    except Exception:
        print(f"Failed to save data due to {Exception}")

def main(args):
    data_root_path = Path(args.data_path)
    out_root_path = Path(args.output_data_path)
    #data_root_path = "/home/pagolpoka/Downloads/medior-data-scientist-case-study"
    data_folder_path = os.path.join(data_root_path, "subjects")
    meta_data_folder_path = os.path.join(data_root_path, "tables")

    subject_folders = os.listdir(data_folder_path)
    meta_file = os.path.join(meta_data_folder_path, 'document_meta.csv')

    df_metadata = pd.read_csv(meta_file)

    dataset_for_training, split_info = split_data_for_training(data_folder_path, subject_folders,args.train_split)
    print(split_info)

    #Have not implemented max token count from argument yet.
    creat_dataset_for_training(dataset_for_training, data_folder_path, df_metadata,out_root_path)
    pass


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--max_token_count', help='Maximum number of words per document', type=int, default=512)
    parser.add_argument('--train_split', help='what percentage of total data to be used for training', type=float, default=.7)
    parser.add_argument('--output_data_path', help='path to the data folder', type=str)
    parser.add_argument('--data_path', help='path to the data folder', type=str)

    main(parser.parse_args())
