import copy
import glob
import json
import pickle
import numpy as np
import sympy
import torch


def float_to_bit_converter(number):
    np_float = np.float16(number)           # convert to float16
    np_float_bytes = np_float.tobytes()     # convert to bytes string \x01\x00\x00\x00\x00\x00...
    np_float_bytes_int = np.frombuffer(np_float_bytes, dtype=np.uint8)  # convert to int list
    np_float_bits_int = np.unpackbits(np_float_bytes_int)       # convert to list of bits
    return np_float_bits_int.tolist()


def unravel_lists(lists) -> list:
    unraveled = []
    for x in lists:
        if isinstance(x, list):
            unraveled.extend(unravel_lists(x))
        else:
            unraveled.append(x)
    return unraveled


# Define a function to recursively convert the SymPy expression to a chain of tokens
def expr_to_tokens(expr):
    converted = []

    if expr.func is not None:
        # test for NumberKind?
        # test for numerator?
        if expr.is_number:
            return str(expr)
        if isinstance(expr, sympy.Symbol):
            return str(expr)
        func_name = expr.func.__name__
        args = [expr_to_tokens(arg2) for arg2 in expr.args]
        converted.append(func_name)
        converted.extend(args)

    expr_args = expr.args
    if len(expr_args) == 0:
        return str(expr)

    return unravel_lists(converted)


def convert_tokens_to_string(token_list):
    equation = ''
    for op in token_list:
        if op == 'Add':
            equation += '+'
        elif op == 'Mul':
            equation += '*'
        elif op == 'Pow':
            equation += '**'
        elif op == 'cosh':
            equation += 'cosh('
        elif op.startswith('C'):
            equation += op
        elif op.startswith('x'):
            equation += op
        elif op == '-1':
            equation += '-'

    equation = equation.replace(')(', ')*(')
    return equation


def clean_tokens(token_list):
    new_token_list = []
    for token in token_list:
        if 'c' in token.strip().lower():
            new_token_list.append('C')
        else:
            new_token_list.append(token)
    return new_token_list


def enhance_with_multi_hot_and_language_tokens(dataset, samples_to_generate, num_samples=50, num_variables=10, enhance_with_half_precision=True):

    x_samples = []
    y_samples = []
    dataset_copy = copy.deepcopy(dataset)

    for idx in range(len(dataset_copy)):

        data = dataset_copy[idx]

        if enhance_with_half_precision:
            x_buffer_array = np.zeros((num_variables, num_samples, 16))
            y_buffer_array = np.zeros((num_samples, 16))

            for var_idx in range(np.asarray(data['x_samples']).shape[0]):

                variables = data['x_samples'][var_idx]

                for sample_idx in range(np.asarray(data['x_samples']).shape[1]):

                    sample = variables[sample_idx]

                    converted = float_to_bit_converter(sample)

                    if len(converted) != 16:
                        print(converted)
                        print(sample)
                        print(variables)
                        print(var_idx, sample_idx)
                        return

                    x_buffer_array[var_idx, sample_idx, :] = converted

            for var_idx in range(np.asarray(data['y_samples']).shape[0]):
                variables = data['y_samples'][var_idx]
                converted = float_to_bit_converter(sample)
                y_buffer_array[var_idx, :] = converted

        constant_counts = data['placeholder'].count('#C')
        constants_numbered = "C{}".join(data['placeholder'].split('#C')).format(*range(constant_counts))
        placeholder_sympy = sympy.parse_expr(constants_numbered)
        tokens = expr_to_tokens(placeholder_sympy)
        tokens_cleaned = clean_tokens(tokens)

        data['tokens'] = tokens
        data['tokens_clean'] = tokens_cleaned

        if enhance_with_half_precision:
            data['x_samples_conv'] = x_buffer_array
            data['y_samples_conv'] = y_buffer_array

        if idx % 100 == 0:
            print('Sampling {}/{}'.format(idx, len(dataset_copy)))

    return dataset_copy


def generate_vocab(token_sequences, vocab_idx_start=1):
    vocab = {}
    vocab_idx = vocab_idx_start

    for token in token_sequences:
        token_strip = token.strip().lower()
        if token_strip not in vocab:
            vocab[token_strip] = vocab_idx
            vocab_idx += 1

    return vocab


def add_start_stop_padding(vocab, eqs_idx):

    old_item = np.array(eqs_idx)
    new_item = np.zeros(len(eqs_idx) + 2)
    # old_item = torch.tensor(eq.numpy())
    # new_item = torch.zeros(old_item.shape[0] + 2)
    new_item[0] = vocab["<START>"]
    new_item[-1] = vocab["<END>"]
    new_item[1:-1] = old_item

    return list(new_item)


def tokens_to_idx(vocab, tokens):
    idx = []
    for x in tokens:
        x_strip = x.strip().lower()
        idx.append(vocab[x_strip])
    # return torch.tensor(idx)
    return idx


def idx_to_tokens(vocab, idx):
    tokens = []
    for x in idx:
        tokens.append(vocab[x])
    return " ".join(tokens)


def generate_vocab_file(dataset, num_variables=10, num_constants=5, replace_constant_index=False, include_padding=False, vocab_provided=None):
    if vocab_provided is None:
        operations = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt', 'add', 'sub', 'mul', 'div', 'pow']
        if include_padding:
            vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        else:
            vocab = {"<START>": 0, "<END>": 1}

        # known mathematical operations, vocab
        start_idx = len(vocab.keys())
        for op in operations:
            op_strip = op.strip().lower()
            if op_strip not in vocab:
                vocab[op_strip] = start_idx
                start_idx += 1

        # known variable and constant names
        start_idx = len(vocab.keys())
        variables = ["x{}".format(i) for i in list(range(0, num_variables))]
        constants = ["c{}".format(i) for i in list(range(0, num_constants))]

        combined = []
        combined.extend(variables)
        if replace_constant_index is False:
            combined.extend(constants)
        else:
            combined.append('C')

        for var in combined:
            var_strip = var.strip().lower()
            if var_strip not in vocab:
                vocab[var_strip] = start_idx
                start_idx += 1
    else:
        vocab = vocab_provided

    for data in dataset:
        if replace_constant_index is True:
            token_input = data['tokens_clean']
        else:
            token_input = data['tokens']
        vocab_tokens = generate_vocab(token_input, vocab_idx_start=len(vocab.keys()))
        for token in vocab_tokens.keys():
            token = token.strip().lower()
            if token not in vocab:
                vocab[token] = len(vocab.keys())

    return vocab


def enhance_with_tokenizer(dataset, vocab, vocab_clean):
    longest_chain_clean = 0
    longest_chain = 0

    for data in dataset:
        vocab_tokens_clean = tokens_to_idx(vocab_clean, data['tokens_clean'])
        vocab_tokens = tokens_to_idx(vocab, data['tokens'])
        padded_tokens_clean = add_start_stop_padding(vocab_clean, vocab_tokens_clean)
        padded_tokens = add_start_stop_padding(vocab, vocab_tokens)
        data['tokenized_clean'] = padded_tokens_clean
        data['tokenized'] = padded_tokens

        if len(padded_tokens_clean) > longest_chain_clean:
            longest_chain_clean = len(padded_tokens_clean)
        if len(padded_tokens) > longest_chain:
            longest_chain = len(padded_tokens)

    return longest_chain, longest_chain_clean


if __name__ == '__main__':
    # enhance_with_multi_hot_and_language_tokens()
    # generate_vocab_file()
    # generate_vocab_file(replace_constant_index=True)
    # enhance_with_tokenizer()

    print()


