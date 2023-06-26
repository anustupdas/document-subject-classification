# Document Subject Text Classification

This repository contains code and supplementary materials which are required to train and evaluate a Hugging face Bert based model
used for text classification task. The task it to classify the subject label given a document.


Codebase:
>  https://github.com/anustupdas/document-sucject-classification.git


Subject datasets:
>  https://drive.google.com/drive/folders/11-dfGZkqZl-LRgo9cCSSsMyVxFgAEMnW?usp=sharing 

Trained Models:
>  https://drive.google.com/drive/folders/11-dfGZkqZl-LRgo9cCSSsMyVxFgAEMnW?usp=sharing 



Fill relevant paths in config.json, and execute the script (--custom_data_path = "Path to the data folder created by prepare_data.py")


## Creating an environment:

    Python Version used: python 3.10.6
    
    !pip install -r requirements.txt

## How to create dataset from training?
    
    

    python prepare_data.py --data_path "You Folder Path to Text files" --output_data_path "Output Dir Path yo Store Training dataframes"

Example:
    
    python prepare_data.py --data_path "/content/drive/MyDrive/medior-data-scientist-case-study" --output_data_path "/content/Training_data"

## How to update training data path in the config file?

    python update_config.py --data_path "Dir to Training dataframes genetrated by prepare_data.py"

Example:
    
    python update_config.py --data_path "/content/Training_data"


## How to run training process?

    python run.py --help

Example:

    !python run.py --custom_data --model --train --epochs 4 --bs 10 --test_bs 8     

## How to evaluate trained model?

    python run.py --custom_data --load_from 'Path To Trained Model .t7 file'

Example:

    python run.py --custom_data --load_from '/content/document-sucject-classification/document-subject-classification-training/document_subject_classification_training/checkpoints/best_model_3.t7'

## How to inference trained model ?

    !python run_inference.py --load_from "Path To Trained Model .t7 file"

Example:

    python run_inference.py --load_from "/content/drive/MyDrive/Rcnn/best_model_14.t7" --text "mon misconception: if critical criminologists state that crime real then they focus on harm (crime real but sufferings are) critical criminology: whole goal is to alleviate the we need to prevent harm, not proper subject is social harm crime catch the harm that happens to groups and we are not only individualizing on criminals but our legal framework also focuses on individual victims, ignoring group and societal crimes physical (example: illness, death from medical system not a crime but causes harm) financial (if we would criminalize being billionaire, criminologists would put all their forces to study but we glorify that emotional (tougher to see and respond pandemic restrictions,..) cultural (access to informational, intellectual, cultural resources (like access to they are par"


## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE.txt) file for details.

## Authors
* **Anustup** anustup.d@rediffmail.com
