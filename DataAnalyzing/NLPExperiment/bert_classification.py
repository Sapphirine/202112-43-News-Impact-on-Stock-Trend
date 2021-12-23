# load packages
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


class Bert_Classifier():
    def __init__(self, model, tokenizer, device, df, label_num=2):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.df = df
        self.label_num = label_num

        # get the train and test dataset
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.25, random_state=1)
        self.x_train, self.y_train = self.get_dataset(self.df_train)
        self.x_test, self.y_test = self.get_dataset(self.df_test)

        self.clf = MLPClassifier(random_state=1, max_iter=600, learning_rate_init=0.0015)
        self.train()



    def truncate_tokenizer(self, text, max_length=512):
        input_ids = self.tokenizer(text, return_tensors='pt', max_length=2048, truncation=True).input_ids
        if input_ids.shape[1] > max_length:
            input_ids = torch.cat((input_ids[:, 0:129], input_ids[:, -383:]), dim=1)
        return input_ids.to(self.device)


    def get_dataset(self, df):
        x = []
        y = []
        with torch.no_grad():
            for i in df.index:
                input_ids = self.truncate_tokenizer(df.loc[i, 'news'])
                outputs = self.model(input_ids=input_ids)
                last_hidden_states = outputs.last_hidden_state
                x.append(last_hidden_states.squeeze()[0].detach().cpu().numpy())
                updown = df.loc[i, '5_day_UpDown_percentage_10_avgline']
                if self.label_num == 3:
                    if updown > 0.02:
                        label = 1
                    elif updown < 0:
                        label = 0
                    else:
                        label = 2
                else:
                    label = 1 if (updown >= 0) else 0
                y.append(label)
        return np.array(x), np.array(y)

    def train(self):
        self.clf.fit(self.x_train, self.y_train)


    def test(self):
        score = self.clf.score(self.x_test, self.y_test)
        print("The accuracy of BERT mode on news {}-classification task is: {}".format(self.label_num, score))
        return




