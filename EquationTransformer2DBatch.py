import glob
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import wandb
from torch.utils import data
import safetensors.torch as safe_torch


class PoolEncoderTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_size):
        super(PoolEncoderTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.lin_attention = nn.Linear(1, self.hidden_size)
        self.attn_softmax = nn.Softmax(dim=2)   # dim 2 = samples
        self.lin_variables_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_variables_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lin_samples = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lin_out = nn.Linear(self.hidden_size * 2, self.embedding_size)

    def extract_by_variable(self, x, is_first=True):
        if is_first:
            x_us = x.unsqueeze(dim=-1)
            x_attn = self.lin_attention(x_us)
            x_attn_sm = self.attn_softmax(x_attn)
            # x_lin = self.lin_variables_1(x.unsqueeze(dim=-1))

            x_lin = self.lin_variables_1(x_attn_sm)
            # test attention with concat
            x_cat = torch.cat((x_lin, x_attn_sm), dim=-1)
            x_lin = self.lin_variables_2(x_cat)
            # test attention with multi
            # x_lin = x_us * x_attn_sm

        else:
            x_lin = self.lin_variables_2(x)

        num_variables = x.shape[1]
        x_max, x_ax_idx = torch.max(x_lin, dim=1)
        x_stack = x_max.unsqueeze(dim=1).repeat(1, num_variables, 1, 1)
        x_cat = torch.cat((x_lin, x_stack), dim=-1)

        return x_cat

    def extract_by_sample(self, x):
        x_lin = self.lin_samples(x)
        num_samples = x.shape[2]
        x_max, x_ax_idx = torch.max(x_lin, dim=2)
        x_stack = x_max.unsqueeze(dim=2).repeat(1, 1, num_samples, 1)
        x_cat = torch.cat((x_lin, x_stack), dim=-1)

        return x_cat

    def forward(self, x):

        # batch x num_var x samples
        for i in range(self.num_layers):
            x = self.extract_by_variable(x, is_first=i == 0)
            # batch x num_var x samples x hidden * 2
            x = self.extract_by_sample(x)
            # batch x num_var x samples x hidden * 2

        # batch x num_var x samples x embedding
        x = self.lin_out(x)
        x_max, x_idx = torch.max(x, dim=2)

        return x_max


class DecoderTransformer(nn.Module):
    def __init__(self, embedding_size, vocab_size, dropout_p, num_heads):
        super(DecoderTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.num_heads = num_heads

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        decoder_layer = nn.TransformerDecoderLayer(self.embedding_size, self.num_heads, batch_first=True)
        self.decoder_module = nn.TransformerDecoder(decoder_layer, 1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lin = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, x, encoder_output):
        x_emb = self.dropout(self.embedding(x))

        x_dec = self.decoder_module(x_emb, encoder_output)

        x_pred = self.lin(x_dec)

        return x_pred


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, encoder: PoolEncoderTransformer, decoder: DecoderTransformer, vocab_size,
                 sample_dist=False, teacher=0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.sample_dist = sample_dist
        self.teacher = teacher


    def forward(self, source, target_set):
        target_set_len = target_set.shape[1]
        batch_size = target_set.shape[0]
        outputs = torch.zeros(batch_size, target_set_len, self.vocab_size).to('cuda')

        encoded = self.encoder(source)

        input_token = target_set[:, 0].to(torch.int32).unsqueeze(dim=1)

        for x in range(1, target_set_len):
            decoded = self.decoder(input_token, encoded)
            next_token = decoded[:, -1, :]
            outputs[:, x, :] = next_token

            if np.random.random() - self.teacher < 0.001:
                input_token = torch.cat((input_token, target_set[:, x].to(torch.int32).unsqueeze(dim=1)), dim=1)
                # input_token = torch.cat((input_token, target[t].to(torch.int32).view(-1)), 0)
            else:
                input_token = torch.cat((input_token, next_token.argmax(dim=1).to(torch.int32).unsqueeze(dim=1)), dim=1)
                # input_token = torch.cat((input_token, output.argmax().to(torch.int32).view(-1)), 0)

        return outputs, input_token


def train_sequence_to_sequence(dataset_to_train, epoch, seq2seq_model: EncoderDecoderTransformer, loss_function,
                               optimizer, scheduler, vocab_size, max_length, batch_size, use_wandb=False, shuffle=True,
                               model_path_saving=None):
    if shuffle:
        random_order = torch.randperm(len(dataset_to_train))
    else:
        random_order = list(range(len(dataset_to_train)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen_tar_seqs = []
    dataset_loss = 0
    losses = []
    for idx in range(0, len(dataset_to_train), batch_size):

        indices = random_order[idx:idx + batch_size]

        # init datasets
        x_samples = torch.tensor([dataset_to_train[x]['x_samples'] for x in indices])
        y_samples = torch.tensor([dataset_to_train[x]['y_samples'] for x in indices])
        target_set = [torch.tensor(dataset_to_train[x]['tokenized_clean']) for x in indices]

        # pad first sequence to max length
        target_set[0] = nn.ConstantPad1d((0, max_length - len(target_set[0])), 0)(target_set[0])
        target_set = torch.nn.utils.rnn.pad_sequence(target_set, batch_first=True, padding_value=0)
        target_set = target_set.to(torch.int32).to(device)

        full_samples = torch.cat((x_samples, y_samples.unsqueeze(dim=1)), dim=1).to(torch.float32)
        full_samples = full_samples.to(device)

        optimizer.zero_grad()

        predicted, tokens = seq2seq_model(full_samples, target_set)

        predicted_sliced = predicted[:, 1:, :]
        target_set_sliced = target_set[:, 1:]

        one_hot_targets = nn.functional.one_hot(target_set_sliced.long(), vocab_size).to(torch.float32)
        # mask padding to be excluded from the loss
        loss = loss_function(predicted_sliced.permute(0, 2, 1), one_hot_targets.permute(0, 2, 1))
        slice_mask = target_set_sliced != 0
        loss = torch.mean(torch.sum(loss * slice_mask.to(torch.float32), dim=1) / torch.count_nonzero(slice_mask, dim=1))

        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if use_wandb is True:
            wandb.log({"Train Loss": loss.item()})
            wandb.log({"Train LossNorm": loss.item() / target_set.shape[1]})

            if scheduler is not None:
                wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})

            if idx % int(1000 / batch_size) == 0:
                gen_seq = torch.argmax(predicted_sliced[0, :, :].detach().to('cpu'), dim=1).numpy()
                tar_seq = target_set_sliced[0, :].detach().to('cpu').numpy()

                gen_tar_seqs.append({"gen": list([int(x) for x in gen_seq]), "tar": list([int(x) for x in tar_seq]),
                                     "idx": int(idx)})

                wandb.log({'Generated Sequence': gen_seq})
                wandb.log({'Expected Sequence': tar_seq})

        losses.append(loss.item())
        if idx % int(100 / batch_size) == 0:
            print('Equation {} Loss: {} LossNorm: {}'.format(idx, loss.item(), loss.item() / target_set.shape[1]))

    if use_wandb is True:
        wandb.log({"Train Loss Mean": np.mean(losses)})

    seq_gen_save_path = model_path_saving + '/train_seq_gen_tar_{}.json'.format(epoch)
    json.dump(gen_tar_seqs, open(seq_gen_save_path, 'w+'), indent=True)

    model_path_to_scan = model_path_saving + '/eqt_*\.pt'
    transformer_saves = len(list(filter(lambda x: 'final' not in x, glob.glob(model_path_to_scan))))
    model_actual_save_path = model_path_saving + '/eqt_{}.pt'.format(transformer_saves)
    torch.save(seq2seq_model.state_dict(), model_actual_save_path)
    # safe_torch.save_file(seq2seq_model.state_dict(), './RNN_LSTM/lstm_{}_final.safetensor'.format(rnn_saves))
    print('Dataset in epoch {} loss: {}'.format(epoch, np.mean(losses)))


def test_sequence_to_sequence(generated_dataset, seq2seq_model, loss_function, batch_size, vocab_size, longest_token_chain, use_wandb=False, type_test='Test'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_avg = 0
    losses = []
    with torch.no_grad():
        non_random_order = list(range(len(generated_dataset)))
        for idx in range(0, len(generated_dataset), batch_size):
            indices = non_random_order[idx:idx + batch_size]

            # init datasets
            x_samples = torch.tensor([generated_dataset[x]['x_samples'] for x in indices])
            y_samples = torch.tensor([generated_dataset[x]['y_samples'] for x in indices])
            target_set = [torch.tensor(generated_dataset[x]['tokenized_clean']) for x in indices]

            # pad first sequence to max length
            target_set[0] = nn.ConstantPad1d((0, longest_token_chain - len(target_set[0])), 0)(target_set[0])
            target_set = torch.nn.utils.rnn.pad_sequence(target_set, batch_first=True, padding_value=0)
            target_set = target_set.to(torch.int32).to(device)

            full_samples = torch.cat((x_samples, y_samples.unsqueeze(dim=1)), dim=1).to(torch.float32)
            full_samples = full_samples.to(device)

            predicted, tokens = seq2seq_model(full_samples, target_set)

            predicted_sliced = predicted[:, 1:, :]
            target_set_sliced = target_set[:, 1:]

            one_hot_targets = nn.functional.one_hot(target_set_sliced.long(), vocab_size).to(torch.float32)
            # loss_test = loss_function(predicted_sliced, one_hot_targets)

            loss = loss_function(predicted_sliced.permute(0, 2, 1), one_hot_targets.permute(0, 2, 1))
            slice_mask = target_set_sliced != 0
            # loss_masked = loss * slice_mask.to(torch.int32)
            # loss = torch.mean(loss_masked)
            loss = torch.mean(torch.sum(loss * slice_mask.to(torch.float32), dim=1) / torch.count_nonzero(slice_mask, dim=1))

            if use_wandb is True:
                wandb.log({"{} Loss".format(type_test): loss.item()})

            losses.append(loss.item())

    if use_wandb is True:
        wandb.log({"{} Loss Mean".format(type_test): np.mean(losses)})

    print('{} Loss Mean: {}'.format(type_test, np.mean(losses)))


def training_run_no_batch_s2s(use_wandb=True, config=None, is_sweep=False, dataset_to_load=None):
    if use_wandb:
        if config is None:
            wandb.init(project='EquationTransformer', settings=wandb.Settings(_service_wait=300))
        else:
            wandb.init(project='EquationTransformer', settings=wandb.Settings(_service_wait=300), config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset_to_load is None:
        data_path = './equation_samples_2d_pad'
    else:
        data_path = './' + dataset_to_load

    data_path_to_make = './EQT_{}'.format(dataset_to_load)
    os.makedirs(data_path_to_make, exist_ok=True)
    model_path_to_make = '{}/EQTransformer'.format(data_path_to_make)
    os.makedirs(model_path_to_make, exist_ok=True)

    vocab_load = json.load(open(data_path + '/token_vocab.json', 'r'))

    batch_size = 25
    num_variables = vocab_load['settings']['num_vars'] + 1
    num_variable_bits = 16
    vocab_size = len(vocab_load['vocab_clean'].keys())
    longest_token_chain = vocab_load['token']['longest_token_chain_clean']
    epochs = 35
    uses_padding = vocab_load['settings']['padding']

    if not is_sweep or not use_wandb:
        hidden_size = 512
        embedding_size = 256
        num_heads = 8
        num_layers = 2
        learning_rate = 0.01
        dropout_rate = 0.05
        sample_size = 1000
        teacher = 0.8
    else:
        hidden_size = wandb.config.hidden_size
        embedding_size = wandb.config.embedding_size
        num_heads = wandb.config.num_heads
        num_layers = wandb.config.num_layers
        learning_rate = wandb.config.learning_rate
        dropout_rate = wandb.config.dropout_rate
        sample_size = 1000
        teacher = wandb.config.teacher

    print('Config: {}'.format(
        [hidden_size, embedding_size, num_heads, num_layers, learning_rate, dropout_rate, sample_size, teacher]))

    # init networks
    encoder = PoolEncoderTransformer(hidden_size=hidden_size, num_layers=num_layers, embedding_size=embedding_size)
    decoder = DecoderTransformer(embedding_size=embedding_size, vocab_size=vocab_size,
                                 dropout_p=dropout_rate, num_heads=num_heads)

    seq2seq_model = EncoderDecoderTransformer(encoder, decoder, vocab_size, teacher=teacher)
    seq2seq_model.to(device)
    # optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(seq2seq_model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=np.ceil(10000 / batch_size) * epochs * len(glob.glob(data_path + '/train/samples_*.pickle'))
    )

    loss_function = nn.CrossEntropyLoss(reduction='none')
    # loss_function = nn.CrossEntropyLoss()

    if use_wandb is True:
        wandb.watch(seq2seq_model)

    for epoch in range(epochs):
        datasets = glob.glob(data_path + '/train/samples_*.pickle')
        for dataset in datasets:
            print('PROCESSING: {}'.format(dataset))
            generated_dataset = pickle.load(open(dataset, 'rb'))
            train_sequence_to_sequence(generated_dataset, epoch, seq2seq_model, loss_function, optimizer, scheduler,
                                       vocab_size, longest_token_chain, batch_size, use_wandb=use_wandb, shuffle=True,
                                       model_path_saving=model_path_to_make)

        # datasets = glob.glob(data_path + '/val/samples_*.pickle')
        # for dataset_path in datasets:
        #     generated_dataset = pickle.load(open(dataset_path, 'rb'))
        #
        #     test_sequence_to_sequence(generated_dataset, seq2seq_model, loss_function, batch_size,
        #                                              vocab_size, longest_token_chain, use_wandb=use_wandb, type_test='Validation')

    datasets = glob.glob(data_path + '/val/samples_*.pickle')
    for dataset_path in datasets:
        generated_dataset = pickle.load(open(dataset_path, 'rb'))

        test_sequence_to_sequence(generated_dataset, seq2seq_model, loss_function, batch_size,
                                                vocab_size, longest_token_chain, use_wandb=use_wandb, type_test='Validation')

    transformer_saves_final = len(glob.glob(model_path_to_make + '/eqt_*_final\.pt'))
    torch.save(seq2seq_model.state_dict(), model_path_to_make + '/eqt_{}_final.pt'.format(transformer_saves_final))


if __name__ == '__main__':

    sweep_config = {
        'method': 'grid',
    }

    sweep_metric = {
        'name': 'loss'
    }

    sweep_parameter = {
        'hidden_size': {
            'values': [512, 1024]
        },
        'embedding_size': {
            'values': [256, 512]
        },
        'num_layers': {
            'values': [1, 2]
        },
        'teacher': {
            'values': [0.3, 0.6, 0.9, 1]
        },
        'num_heads': {
            'values': [4, 8]
        }
    }

    sweep_config['metric'] = sweep_metric
    sweep_config['parameters'] = sweep_parameter

    wandb.login('allow', '<WEIGHTS AND BIASES KEY HERE>')

    datasets = ['equation_samples_500_2',
                'equation_samples_500_3',
                'equation_samples_500_5',
                'equation_samples_500_10',
                'equation_samples_1000_2',
                'equation_samples_1000_3',
                'equation_samples_1000_5',
                'equation_samples_1000_10']

    for set_name in datasets:
        training_run_no_batch_s2s(use_wandb=False, is_sweep=False, dataset_to_load=set_name)
