from argparse import ArgumentParser

from transformers import BertTokenizer
from constants import DEFAULT_TOKENIZER_TYPE
import torch
label_dict = {"Accounting": 0, "Aerospace Engineering": 1, "Agriculture": 2, "Algebra": 3, "Anthropology": 4, "Architecture": 5, "Astronomy": 6, "Biology": 7, "Calculus": 8, "Chemical Engineering": 9, "Chemistry": 10, "Civil Engineering": 11, "Communication Science": 12, "Computer Science": 13, "Criminology": 14, "Culinary Arts": 15, "Dentistry": 16, "Earth Science": 17, "Econometrics": 18, "Economics": 19, "Educational Science": 20, "Electrical Engineering": 21, "English": 22, "Entrepreneurship": 23, "Environmental Science": 24, "Finance": 25, "Food Science": 26, "French": 27, "Geography": 28, "Geological Science": 29, "Geometry": 30, "History": 31, "Industrial Design": 32, "Industrial Engineering": 33, "Law": 34, "Linguistics": 35, "Literature": 36, "Logic": 37, "Management": 38, "Mechanical Engineering": 39, "Medicine": 40, "Music": 41, "Nursing": 42, "Performing Arts": 43, "Philosophy": 44, "Physics": 45, "Political Science": 46, "Probability": 47, "Psychology": 48, "Public Administration": 49, "Religious Studies": 50,"Sociology": 51, "Spanish": 52, "Statistics": 53, "Trigonometry": 54, "Visual Arts": 55}


def predict(test_input, model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    mask = test_input['attention_mask'].to(device)
    input_id = test_input['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask)
    predicted_label = output.argmax(dim=1).tolist()[0]
    class_label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]
    print(class_label)

def main(args):

    tokenizer = BertTokenizer.from_pretrained(DEFAULT_TOKENIZER_TYPE)
    text = "It is Very hot in Assam"
    targets_labels = ["Accounting", "Aerospace Engineering", "Agriculture", "Algebra", "Anthropology", "Architecture",
                      "Astronomy", "Biology", "Calculus", "Chemical Engineering", "Chemistry", "Civil Engineering",
                      "Communication Science", "Computer Science", "Criminology", "Culinary Arts", "Dentistry",
                      "Earth Science", "Econometrics", "Economics", "Educational Science", "Electrical Engineering",
                      "English", "Entrepreneurship", "Environmental Science", "Finance", "Food Science", "French",
                      "Geography", "Geological Science", "Geometry", "History", "Industrial Design",
                      "Industrial Engineering", "Law", "Linguistics", "Literature", "Logic", "Management",
                      "Mechanical Engineering", "Medicine", "Music", "Nursing", "Performing Arts", "Philosophy",
                      "Physics", "Political Science", "Probability", "Psychology", "Public Administration",
                      "Religious Studies", "Sociology", "Spanish", "Statistics", "Trigonometry", "Visual Arts"]
    if args.text:
        text = args.text

    encoded_text = tokenizer(str(text), padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    if not args.load_from:
        print("Please provide model_path to load from for inferencing")
        return


    print("Model_Path: ", args.load_from)
    with open(args.load_from, 'rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))

    predict(encoded_text, model)




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--input_directory', help='folder to the text files to be inferred',default=None, type=str)
    parser.add_argument('--input_file', help='path to the text file to be inferred', default=None, type=str)
    parser.add_argument('--text', help='raw text as input', default="Artificial intelligence in computer science are "
                                                                    "achiving new heights", type=str)



    main(parser.parse_args())
