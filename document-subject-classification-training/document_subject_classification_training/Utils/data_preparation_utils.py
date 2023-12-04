import os
import re
import nltk
import pandas as pd
from Utils.utils import prepare_output_dir
from Utils.data_preparation_utils import *
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def drop_empty_sections(section_list):

    for idx,sec in  enumerate(section_list):

        if len(sec) < 10:
            del section_list[idx]
    return section_list

def count_words(text):

    return len(nltk.word_tokenize(text))


def get_word_count(file_name,df_metadata):
    file_name = int(file_name[:-4])
    word_count = 0
    if file_name in df_metadata['document_id'].values:
        idx = list(df_metadata['document_id'].values).index(file_name)
        word_count = df_metadata['word_count'][idx]
    return word_count


def get_summary(file_name,df_metadata):
    summary = ""
    file_name = int(file_name[:-4])
    if file_name in df_metadata['document_id'].values:
        idx = list(df_metadata['document_id'].values).index(file_name)
        summary = df_metadata['summary'][idx]
    return summary


def get_seperator_foramt(levels=None):
    level_format = '\d' if levels == None else '[' + str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_format = ',' + level_format + ",.*?\."
    return seperator_format


def preprocess_content_text(raw_text):
    pattern_to_omit = get_seperator_foramt((3, 999))
    cleaned = re.sub(pattern_to_omit, "", raw_text).lower()
    cleaned = re.sub("_", "", cleaned)
    cleaned = re.sub("/n", "", cleaned)
    return cleaned


def break_text_in_parts(content, word_limit=510):
    sec_list = []
    stop_words = set(stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish'))
    #discard all the empty files
    if len(content) > 0:
        clean_text = preprocess_content_text(content[0])
        sentence = nltk.sent_tokenize(clean_text)

        # print(len(sentence))
        section_word_count = 0
        section_text = ' '
        for idx, st in enumerate(sentence):

            if len(st) > 10:

                words = nltk.word_tokenize(st)

                section_word_count += len(words)
                text = ' '.join([word for word in words if word not in stop_words])
                if word_limit > section_word_count :
                    section_text += "" + text

                else:

                    sec_list.append(section_text)
                    section_word_count = len(words)
                    section_text = st
        sec_list.append(section_text)
    sec_list = drop_empty_sections(sec_list)
    return sec_list


def save_datasets(dataset_dataframe, output_path):
    print(len(dataset_dataframe))
    train_df = pd.DataFrame(dataset_dataframe[0], columns=['file_name', 'text', 'accept', 'category'])
    val_df = pd.DataFrame(dataset_dataframe[1], columns=['file_name', 'text', 'accept', 'category'])
    test_df = pd.DataFrame(dataset_dataframe[2], columns=['file_name', 'text', 'accept', 'category'])

    prepare_output_dir(output_path)

    train_df.to_csv(os.path.join(output_path,'train/token_main_data_subject_classification_summary_Train.csv'))
    val_df.to_csv(os.path.join(output_path,'dev/token_main_data_subject_classification_summary_Val.csv'))
    test_df.to_csv(os.path.join(output_path,'test/token_main_data_subject_classification_summary_Test.csv'))