import glob
import json
import pickle
import os
import EquationGenerator
import DataEnrichment


def generate_multiple_set(train_set_count, test_set_count, val_set_count, file_path__reference):
    samples_to_generate = 10000
    equation_length = 5
    num_variables = 10
    num_constants = 5
    constant_chance = 0.3
    sample_size = 1000
    equation_min_max = 100
    min_value = -30
    max_value = 30
    equation_sample_method = 'random'
    use_padding = True
    enhance_with_half_precision = False
    sample_per_variable = True

    settings = {
        'samples': samples_to_generate,
        'eq_length': equation_length,
        'num_vars': num_variables,
        'num_constants': num_constants,
        'c_chance': constant_chance,
        'sample_size': sample_size,
        'eq_min_max': equation_min_max,
        'min_value': min_value,
        'max_value': max_value,
        'sample_method': equation_sample_method,
        'padding': use_padding,
        'half-precision': enhance_with_half_precision,
        'sample_per_variable': sample_per_variable
    }

    filepath = file_path__reference
    os.makedirs(filepath, exist_ok=True)

    for x in range(train_set_count):
        generated_dataset = EquationGenerator.generate_dataset(samples_to_generate, constant_chance=constant_chance,
                                                               pool_size=5, chunk_size=1000, sample_time=0.1, timeout=30,
                                                               sample_size=sample_size, num_variables=num_variables,
                                                               num_constants=num_constants, equation_length=equation_length,
                                                               equation_min_max=equation_min_max, min_value=min_value, max_value=max_value,
                                                               sample_method=equation_sample_method, sample_per_variable=sample_per_variable)

        generated_dataset_copy = DataEnrichment.enhance_with_multi_hot_and_language_tokens(generated_dataset,
                                                                                           samples_to_generate,
                                                                                           num_samples=sample_size,
                                                                                           num_variables=num_variables,
                                                                                           enhance_with_half_precision=enhance_with_half_precision)

        filepath_train = file_path__reference + '/train'
        os.makedirs(filepath_train, exist_ok=True)
        with open('./{}/samples_{}__{}.pickle'.format(filepath_train, samples_to_generate, x), 'wb+') as file:
            pickle.dump(generated_dataset_copy, file)

    for x in range(test_set_count):
        generated_dataset = EquationGenerator.generate_dataset(samples_to_generate, constant_chance=constant_chance,
                                                               pool_size=5, chunk_size=1000, sample_time=0.1, timeout=30,
                                                               sample_size=sample_size, num_variables=num_variables,
                                                               num_constants=num_constants, equation_length=equation_length,
                                                               equation_min_max=equation_min_max,
                                                               sample_method=equation_sample_method)

        generated_dataset_copy = DataEnrichment.enhance_with_multi_hot_and_language_tokens(generated_dataset,
                                                                                           samples_to_generate,
                                                                                           num_samples=sample_size,
                                                                                           num_variables=num_variables,
                                                                                           enhance_with_half_precision=enhance_with_half_precision)
        filepath_test = file_path__reference + '/test'
        os.makedirs(filepath_test, exist_ok=True)
        with open('./{}/samples_{}__{}.pickle'.format(filepath_test, samples_to_generate, x), 'wb+') as file:
            pickle.dump(generated_dataset_copy, file)

    for x in range(val_set_count):
        generated_dataset = EquationGenerator.generate_dataset(samples_to_generate, constant_chance=constant_chance,
                                                               pool_size=5, chunk_size=1000, sample_time=0.1, timeout=30,
                                                               sample_size=sample_size, num_variables=num_variables,
                                                               num_constants=num_constants, equation_length=equation_length,
                                                               equation_min_max=equation_min_max,
                                                               sample_method=equation_sample_method)

        generated_dataset_copy = DataEnrichment.enhance_with_multi_hot_and_language_tokens(generated_dataset,
                                                                                           samples_to_generate,
                                                                                           num_samples=sample_size,
                                                                                           num_variables=num_variables,
                                                                                           enhance_with_half_precision=enhance_with_half_precision)
        filepath_val = file_path__reference + '/val'
        os.makedirs(filepath_val, exist_ok=True)
        with open('./{}/samples_{}__{}.pickle'.format(filepath_val, samples_to_generate, x), 'wb+') as file:
            pickle.dump(generated_dataset_copy, file)

    vocab_file = None
    vocab_file_clean = None
    longest_token_chain = 0
    longest_token_chain_clean = 0

    train_sets = glob.glob('./{}/train/samples_{}__*.pickle'.format(file_path__reference, samples_to_generate))
    test_sets = glob.glob('./{}/test/samples_{}__*.pickle'.format(file_path__reference, samples_to_generate))
    val_sets = glob.glob('./{}/val/samples_{}__*.pickle'.format(file_path__reference, samples_to_generate))
    combined = []
    combined.extend(train_sets)
    combined.extend(test_sets)
    combined.extend(val_sets)
    for pickled in combined:
        dataset_loaded = pickle.load(open(pickled, 'rb'))

        vocab_file = DataEnrichment.generate_vocab_file(dataset_loaded,
                                                        num_variables=num_variables,
                                                        num_constants=num_constants,
                                                        include_padding=use_padding,
                                                        vocab_provided=vocab_file)
        vocab_file_clean = DataEnrichment.generate_vocab_file(dataset_loaded,
                                                              num_variables=num_variables,
                                                              num_constants=num_constants,
                                                              include_padding=use_padding,
                                                              replace_constant_index=True,
                                                              vocab_provided=vocab_file_clean)

        longest_token_chain, longest_token_chain_clean = DataEnrichment.enhance_with_tokenizer(dataset_loaded,
                                                                                               vocab_file,
                                                                                               vocab_file_clean)

        with open(pickled, 'wb+') as file:
            pickle.dump(dataset_loaded, file)

    if use_padding:
        for pickled in combined:
            dataset_loaded = pickle.load(open(pickled, 'rb'))
            # append 0s until max chain length

    json.dump(
        {"token":
            {
                'longest_token_chain': longest_token_chain,
                'longest_token_chain_clean': longest_token_chain_clean
            },
            "vocab": vocab_file,
            "vocab_clean": vocab_file_clean,
            "settings": settings
        },
        open('./{}/token_vocab.json'.format(filepath), 'w+'), indent=True)


if __name__ == '__main__':
    generate_multiple_set(3, 1, 1, 'equation_samples_1000_10')
