import os
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def train(model, train_data, val_data, learning_rate, epochs, checkpoint_path):

    #train, val = Dataset(train_data), Dataset(val_data)

    #train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    train_dataloader = train_data
    val_dataloader = val_data

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    best_val_pk = 0
    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        eval_acc = total_acc_val / len(val_data)
        if eval_acc > best_val_pk:
            best_val_pk = eval_acc

            save_path = os.path.join(checkpoint_path, f'best_model_{epoch_num}.t7')
            print(f"Saving Best model at epoch {epoch_num}: {save_path}")
            torch.save(model, save_path)

        # save_path = os.path.join(checkpoint_path, f'model_{epoch_num}.t7')
        # print("Saving: ", save_path)
        # torch.save(model, save_path)




def evaluate(model, test_data):
    print("Evaluating")

    pred_list = []
    original_list = []
    targets_labels = ["Accounting", "Aerospace Engineering", "Agriculture", "Algebra", "Anthropology", "Architecture", "Astronomy", "Biology", "Calculus", "Chemical Engineering", "Chemistry", "Civil Engineering", "Communication Science", "Computer Science", "Criminology", "Culinary Arts", "Dentistry", "Earth Science", "Econometrics", "Economics", "Educational Science", "Electrical Engineering", "English", "Entrepreneurship", "Environmental Science", "Finance", "Food Science", "French", "Geography", "Geological Science", "Geometry", "History", "Industrial Design", "Industrial Engineering", "Law", "Linguistics", "Literature", "Logic", "Management", "Mechanical Engineering", "Medicine", "Music", "Nursing", "Performing Arts", "Philosophy", "Physics", "Political Science", "Probability", "Psychology", "Public Administration", "Religious Studies","Sociology", "Spanish", "Statistics", "Trigonometry", "Visual Arts"]

    test_dataloader = test_data

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            pred_list.append(output.argmax(dim=1).tolist()[0])
            original_list.append(test_label.tolist()[0])
            #print("Origival vs Predicted: ", test_label.tolist()[0], output.argmax(dim=1).tolist()[0])

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    y_pred = np.array(pred_list)
    y_test = np.array(original_list)

    print(f"Pred: {y_pred}")
    print(f"Original: {y_test}")

    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=targets_labels))

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return total_acc_test / len(test_data)
