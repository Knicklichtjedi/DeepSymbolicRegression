import glob
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils import data
import safetensors.torch as safe_torch


class EncoderLSTM(nn.Module):
    def __init__(self, variables, variable_bits, batch_size, hidden_size, num_layers, max_length, attention):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.variables = variables
        self.variable_bits = variable_bits
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.attention = attention

        if batch_size != 0:
            self.lstm_1 = nn.LSTM(input_size=self.variable_bits, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            if self.attention:
                self.lstm_2 = nn.LSTM(input_size=self.variables, hidden_size=self.max_length, num_layers=self.num_layers, batch_first=True)
        else:
            self.lstm_1 = nn.LSTM(input_size=self.variable_bits, hidden_size=self.hidden_size, num_layers=self.num_layers)
            if self.attention:
                self.lstm_2 = nn.LSTM(input_size=self.variables, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.lin = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden_1, concat_1, hidden_2=None, concat_2=None):
        # x_red = torch.squeeze(self.lin(x))
        output, (hidden_1, concat_1) = self.lstm_1(x, (hidden_1, concat_1))

        if self.attention:
            output, (hidden_2, concat_2) = self.lstm_2(output.rot90(), (hidden_2, concat_2))
            output = self.lin(output)
            return output, (hidden_1, concat_1, hidden_2, concat_2)
        else:
            return output, (hidden_1, concat_1)

    def init_hidden_concat_lstm(self):
        # num layers x hidden dim
        if self.batch_size != 0:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to('cuda'), \
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to('cuda')
        else:
            return torch.zeros(self.num_layers, self.hidden_size).to('cuda'), \
                torch.zeros(self.num_layers, self.hidden_size).to('cuda')


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, embedding_size, vocab_size, sample_size, dropout_p, attention):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.attention = attention
        self.sample_size = sample_size

        if self.attention:
            self.embedding = nn.Embedding(vocab_size, self.hidden_size)
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
            # self.attn_combine = nn.Linear(self.hidden_size * (1 + self.num_layers), self.hidden_size)
            self.attn_combine = nn.Linear(2, 1)    # sample size + embedding
            self.embedding_size = hidden_size
            embedding_size = hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        self.lin = nn.Linear(hidden_size, vocab_size)
        if self.batch_size != 0:
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.attn_softmax = nn.Softmax(dim=1)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
            self.attn_softmax = nn.Softmax(dim=1)
            self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, hidden, concat, encoder_output=None):
        embedded = self.dropout(self.embedding(x))
        # Nx256

        if self.attention:
            attn_weights = self.attn_softmax(self.attn(hidden))
            attn_applied = attn_weights * encoder_output.T
            output = torch.cat((embedded, attn_applied))
            output = self.attn_combine(output.T)
            embedded = F.relu(output).T

        output, (hidden, concat) = self.lstm(embedded, (hidden, concat))
        output = torch.squeeze(output)
        logits = self.lin(output)
        return logits, (hidden, concat)

    def init_hidden_concat_lstm(self):
        # num layers x hidden dim
        if self.batch_size != 0:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to('cuda'), \
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to('cuda')
        else:
            return torch.zeros(self.num_layers, self.hidden_size).to('cuda'), \
                torch.zeros(self.num_layers, self.hidden_size).to('cuda')


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM, vocab_size, sample_dist=False, teacher=False, attention=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.sample_dist = sample_dist
        self.teacher = teacher
        self.attention = attention

        self.softmax = nn.Softmax(dim=0)

    def forward(self, source, target):
        target_len = target.shape[0]
        outputs = torch.zeros(target_len, self.vocab_size).to('cuda')

        encoder_hidden_1, encoder_concat_1 = self.encoder.init_hidden_concat_lstm()
        if self.attention:
            encoder_hidden_2, encoder_concat_2 = self.encoder.init_hidden_concat_lstm()
            encoder_outputs = torch.zeros(source.shape[1], self.encoder.hidden_size).to('cuda')

        for sample in range(source.shape[1]):
            if self.attention:
                encoder_output, states = self.encoder(source[:, sample, :], encoder_hidden_1, encoder_concat_1, encoder_hidden_2, encoder_concat_2)
                encoder_hidden_1, encoder_concat_1, encoder_hidden_2, encoder_concat_2 = states
                encoder_outputs[sample, :] = encoder_output.squeeze()
            else:
                encoder_output, states = self.encoder(source[:, sample, :], encoder_hidden_1, encoder_concat_1)
                encoder_hidden_1, encoder_concat_1 = states

        if self.attention:
            decoder_hidden = encoder_hidden_2
            decoder_concat = encoder_concat_2
        else:
            decoder_hidden = encoder_hidden_1
            decoder_concat = encoder_concat_1

        input_token = target[0].to(torch.int32).view(-1)
        for t in range(1, target_len):
            if self.attention:
                output, (decoder_hidden, decoder_concat) = self.decoder(input_token, decoder_hidden, decoder_concat, encoder_output)
            else:
                output, (decoder_hidden, decoder_concat) = self.decoder(input_token, decoder_hidden, decoder_concat)
            outputs[t, :] = output.view(-1, self.vocab_size)[-1, :]

            if isinstance(self.teacher, bool):
                if self.teacher:
                    input_token = target[t].to(torch.int32).unsqueeze(dim=0)
                else:
                    input_token = output.argmax().to(torch.int32).unsqueeze(dim=0)
            elif isinstance(self.teacher, float):
                teacher_chance = np.random.random()
                if teacher_chance - self.teacher <= 0.001:
                    input_token = target[t].to(torch.int32).unsqueeze(dim=0)
                else:
                    input_token = output.argmax().to(torch.int32).unsqueeze(dim=0)
            else:
                raise Exception("Not a compatible teacher type {}, expected bool/float.".format(type(self.teacher)))
        return outputs


def train_sequence_to_sequence(dataset_to_train, epoch, seq2seq_model: EncoderDecoderLSTM, loss_function, optimizer, vocab_size, use_wandb=False, shuffle=True):
    if shuffle:
        random_order = torch.randperm(len(dataset_to_train))
    else:
        random_order = list(range(len(dataset_to_train)))

    dataset_loss = 0
    for idx in random_order:
        data_point = dataset_to_train[idx]

        # init datasets
        x_samples = data_point['x_samples_conv']
        y_samples = data_point['y_samples_conv']

        full_samples = torch.tensor(np.concatenate((x_samples, np.expand_dims(y_samples, axis=0)), axis=0),
                                    dtype=torch.float32)
        full_samples = full_samples.to('cuda')

        target_set = torch.tensor(data_point['tokenized_clean']).to(torch.int32).to('cuda')

        optimizer.zero_grad()

        input_set = full_samples[:, :, :]
        predicted = seq2seq_model(input_set, target_set)

        predicted_sliced = predicted[1:].view(-1, predicted.shape[-1])
        target_set_sliced = target_set[1:].view(-1)

        one_hot_targets = nn.functional.one_hot(target_set_sliced.long(), vocab_size).to(torch.float32)
        loss = loss_function(predicted_sliced, one_hot_targets)

        loss.backward()
        optimizer.step()

        if use_wandb is True:
            wandb.log({"Loss": loss.item()})

            if idx % 1000 == 0:
                gen_seq = torch.argmax(predicted_sliced.detach().to('cpu'), dim=1)
                tar_seq = target_set_sliced.detach().to('cpu')
                wandb.log({'Generated Sequence': gen_seq})
                wandb.log({'Expected Sequence': tar_seq})

        dataset_loss += loss.item()
        if idx % 1000 == 0:
            print('Equation {} Loss: {}'.format(idx, loss))

    dataset_loss = dataset_loss / len(dataset_to_train)
    if use_wandb is True:
        wandb.log({"Epoch Loss": dataset_loss})

    rnn_saves = len(list(filter(lambda x: 'final' not in x, glob.glob('./RNN_LSTM/lstm_*.pt'))))
    torch.save(seq2seq_model.state_dict(), './RNN_LSTM/lstm_{}.pt'.format(rnn_saves))
    print('Epoch {} loss: {}'.format(epoch, dataset_loss))


def training_run_no_batch_s2s(use_wandb=True, config=None):
    if use_wandb:
        if config is None:
            wandb.init(project='EquationTransformer', settings=wandb.Settings(_service_wait=300))
        else:
            wandb.init(project='EquationTransformer', settings=wandb.Settings(_service_wait=300), config=config)

    vocab_load = json.load(open('./equation_samples/token_vocab.json', 'r'))

    batch_size = 0
    num_variables = 3 + 1
    num_variable_bits = 16
    vocab_size = len(vocab_load['vocab_clean'].keys())
    longest_token_chain = vocab_load['token']['longest_token_chain_clean']

    if not use_wandb:
        hidden_size = 256
        embedding_size = 64
        num_layers = 1
        learning_rate = 0.01
        dropout_rate = 0.05  # reintroduce later
        sample_size = 1000
        teacher = True
        attention = True
    else:
        hidden_size = wandb.config.hidden_size
        num_layers = wandb.config.num_layers
        embedding_size = wandb.config.embedding_size
        learning_rate = wandb.config.learning_rate
        dropout_rate = wandb.config.dropout_rate
        teacher = wandb.config.teacher
        attention = wandb.config.attention
        sample_size = 1000

    print('Config: {}'.format([hidden_size, embedding_size, num_layers, learning_rate, dropout_rate, sample_size, teacher, attention]))

    # init networks
    encoder = EncoderLSTM(variables=num_variables, variable_bits=num_variable_bits, batch_size=batch_size,
                          hidden_size=hidden_size, num_layers=num_layers, max_length=longest_token_chain,
                          attention=attention)
    decoder = DecoderLSTM(input_size=1, embedding_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size,
                          num_layers=num_layers, vocab_size=vocab_size, sample_size=sample_size,
                          dropout_p=dropout_rate, attention=attention)

    seq2seq_model = EncoderDecoderLSTM(encoder, decoder, vocab_size, teacher=teacher, attention=attention)
    seq2seq_model.to('cuda')
    optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    if use_wandb is True:
        wandb.watch(seq2seq_model)

    generated_dataset = pickle.load(open('./equation_samples/samples_conv_10000.pickle', 'rb'))[:500]
    train_sequence_to_sequence(generated_dataset, 0, seq2seq_model, loss_function, optimizer, vocab_size, use_wandb=use_wandb)

    rnn_saves_final = len(glob.glob('./RNN_LSTM/lstm_*_final\.pt'))
    torch.save(seq2seq_model.state_dict(), './RNN_LSTM/lstm_{}_final.pt'.format(rnn_saves_final))


if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
    }

    sweep_metric = {
        'name': 'loss'
    }

    sweep_parameter = {
        'hidden_size': {
            'values': [512, 256, 128]
        },
        'embedding_size': {
            'values': [128, 64, 32]
        },
        'attention': {
            'values': [True, False]
        },
        'num_layers': {
            'values': [1]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'dropout_rate': {
            'values': [0.05]
        },
        'teacher': {
            'values': [True]
        }
    }

    sweep_config['metric'] = sweep_metric
    sweep_config['parameters'] = sweep_parameter

    # training_run_no_batch_ED()
    # training_run_no_batch_s2s()

    # all trainable
    # [np.prod(p.size()) for p in filter(lambda p: p.requires_grad, seq2seq_model.parameters())]

    # all layers
    # list(seq2seq_model.named_modules())

    # wandb.login('allow', '<WEIGHTS AND BIASES KEY HERE>')
    # sweep_id = wandb.sweep(sweep=sweep_config, project='EquationTransformer')
    # wandb.agent(sweep_id, function=training_run_no_batch_s2s)

    training_run_no_batch_s2s(use_wandb=False)



