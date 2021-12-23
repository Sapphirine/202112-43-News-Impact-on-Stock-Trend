# load packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import sentencepiece
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import re
from typing import Dict, Any


class T5_Classifier():
    def __init__(self, model, tokenizer, device, df, label_num=2):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.df = df
        self.label_num = label_num

        # get the train and test dataset
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.25, random_state=1)
        self.dataset_train = self.get_dataset(self.df_train)
        self.dataset_test = self.get_dataset(self.df_test)

        self.train_dataloader = DataLoader(self.dataset_train,
                                           batch_size=12,
                                           # sampler=train_sampler,
                                           shuffle=False)

        # self.valid_dataloader = DataLoader(self.dataset_dev,
        #                               batch_size=24,
        #                               shuffle=True)

        self.test_dataloader = DataLoader(self.dataset_test,
                                          batch_size=12,
                                          shuffle=True)

    def tokenize_corpus(self, df, max_len=1024):
        # token ID storage
        input_ids = []
        # attension mask storage
        attention_masks = []
        # max len -- 512 is max
        max_len = max_len
        # for every document:
        for doc in df:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                doc,  # document to encode.
                add_special_tokens=True,  # add tokens relative to model
                max_length=max_len,  # set max length
                truncation=True,  # truncate longer messages
                pad_to_max_length=True,  # add padding
                return_attention_mask=True,  # create attn. masks
                return_tensors='pt'  # return pytorch tensors
            )

            # add the tokenized sentence to the list
            input_ids.append(encoded_dict['input_ids'])

            # and its attention mask (differentiates padding from non-padding)
            attention_masks.append(encoded_dict['attention_mask'])
        return torch.cat(input_ids, dim=0).to(self.device), torch.cat(attention_masks, dim=0).to(self.device)

    def get_dataset(self, df):
        text_input_ids, text_attention_masks = self.tokenize_corpus(df['news'].values)
        targets = []
        cnt_pos = 0
        cnt_neg = 0
        cnt_neu = 0
        for i in df.index:
            updown = df.loc[i, '5_day_UpDown_percentage_10_avgline']
            if self.label_num == 3:
                if updown > 0.02:
                    targets.append('positive')
                    cnt_pos += 1
                elif updown < 0:
                    targets.append('negative')
                    cnt_neg += 1
                else:
                    targets.append('neutral')
                    cnt_neu += 1
            else:
                targets.append('positive' if (updown >= 0) else 'negative')
        target_input_ids, target_attention_masks = self.tokenize_corpus(targets, 3)
        dataset = TensorDataset(text_input_ids, text_attention_masks, target_input_ids, target_attention_masks)
        return dataset

    def train(self):

        # Adam w/ Weight Decay Fix
        # set to optimizer_grouped_parameters or model.parameters()
        self.optimizer = AdamW(self.model.parameters(),
                          lr=8e-5
                          )

        # epochs
        self.epochs = 8

        # lr scheduler
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        # create gradient scaler for mixed precision
        self.scaler = GradScaler()

        self.training_stats = []
        self.valid_stats = []
        self.best_valid_loss = float('inf')

        # for each epoch
        for epoch in range(self.epochs):
            # train
            self.training(self.train_dataloader, epoch)


    def test(self):
        return self.testing(self.test_dataloader)


    def training(self, dataloader, epoch):

        # capture time
        total_t0 = time.time()

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.epochs))
        print('Training...')

        # reset total loss for epoch
        train_total_loss = 0
        total_train_f1 = 0

        # put model into traning mode
        self.model.train()

        # for each batch of training data...
        for step, batch in enumerate(dataloader):
            # progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

            # Unpack this training batch from our dataloader:
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input tokens
            #   [1]: attention masks
            #   [2]: target tokens
            #   [3]: target attenion masks
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_target_ids = batch[2]
            b_target_mask = batch[3]

            # clear previously calculated gradients
            self.optimizer.zero_grad()

            # runs the forward pass with autocasting.
            with autocast():
                # forward propagation (evaluate model on training batch)
                outputs = self.model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_target_ids,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]

                # sum the training loss over all batches for average loss at end
                # loss is a tensor containing a single value
                train_total_loss += loss.item()

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

            # update the learning rate
            self.scheduler.step()

        # calculate the average loss over all of the batches
        avg_train_loss = train_total_loss / len(dataloader)

        # Record all statistics from this epoch.
        self.training_stats.append(
            {
                'Train Loss': avg_train_loss
            }
        )

        # training time end
        training_time = self.format_time(time.time() - total_t0)

        # print result summaries
        print("")
        print("summary results")
        print("epoch | trn loss | trn time ")
        print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")



    def validating(self, dataloader):

        # capture validation time
        total_t0 = time.time()

        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")

        # put the model in evaluation mode
        self.model.eval()

        # track variables
        total_valid_loss = 0

        # evaluate data for one epoch
        for batch in dataloader:
            # Unpack this training batch from our dataloader:
            # `batch` contains three pytorch tensors:
            #   [0]: input tokens
            #   [1]: attention masks
            #   [2]: target tokens
            #   [3]: target attenion masks
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_target_ids = batch[2]
            b_target_mask = batch[3]

            # tell pytorch not to bother calculating gradients
            # as its only necessary for training
            with torch.no_grad():
                # forward propagation (evaluate model on training batch)
                outputs = self.model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_target_ids,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]

                # sum the training loss over all batches for average loss at end
                # loss is a tensor containing a single value
                total_valid_loss += loss.item()

        # calculate the average loss over all of the batches.
        global avg_val_loss
        avg_val_loss = total_valid_loss / len(dataloader)

        # Record all statistics from this epoch.
        self.valid_stats.append(
            {
                'Val Loss': avg_val_loss,
                'Val PPL.': np.exp(avg_val_loss)
            }
        )

        # capture end validation time
        training_time = self.format_time(time.time() - total_t0)

        # print result summaries
        print("")
        print("summary results")
        print("epoch | val loss | val ppl | val time")
        print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {np.exp(avg_val_loss):.3f} | {training_time:}")


    def testing(self, dataloader):

        print("")
        print("Running Testing...")

        # measure training time
        t0 = time.time()

        # put the model in evaluation mode
        self.model.eval()

        # track variables
        total_test_loss = 0
        total_test_acc = 0
        total_test_f1 = 0
        predictions = []
        actuals = []
        test_stats = []

        # evaluate data for one epoch
        for step, batch in enumerate(dataloader):
            # progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = self.format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # Unpack this training batch from our dataloader:
            # `batch` contains three pytorch tensors:
            #   [0]: input tokens
            #   [1]: attention masks
            #   [2]: target tokens
            #   [3]: target attenion masks
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_target_ids = batch[2]
            b_target_mask = batch[3]

            # tell pytorch not to bother calculating gradients
            # as its only necessary for training
            with torch.no_grad():

                # forward propagation (evaluate model on training batch)
                outputs =self. model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_target_ids,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]
                total_test_loss += loss.item()

                decoder_start_token_id = 0
                decoder_input_ids = (
                        torch.ones((b_input_ids.shape[0], 1), dtype=torch.long,
                                   device=b_input_ids.device) * decoder_start_token_id
                )
                generated_ids = self.model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3,
                    # decoder_input_ids = decoder_input_ids
                )

                preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                          b_target_ids]

                total_test_acc += accuracy_score(target, preds)
                total_test_f1 += f1_score(preds, target,
                                          average='weighted',
                                          labels=np.unique(preds))
                predictions.extend(preds)
                actuals.extend(target)

        # calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(dataloader)

        avg_test_acc = total_test_acc / len(dataloader)

        avg_test_f1 = total_test_f1 / len(dataloader)

        # Record all statistics from this epoch.
        test_stats.append(
            {
                'Test Loss': avg_test_loss,
                'Test PPL.': np.exp(avg_test_loss),
                'Test Acc.': avg_test_acc,
                'Test F1': avg_test_f1
            }
        )
        temp_data = pd.DataFrame({'predicted': predictions, 'actual': actuals})

        return test_stats, temp_data

    # time function
    def format_time(self,elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))



