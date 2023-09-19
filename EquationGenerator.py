import multiprocessing
import pickle
import time
import warnings
from collections import OrderedDict
from typing import Tuple
from functools import partial
import pathos
import numpy as np
import sympy as sympy
from sympy.utilities.exceptions import ignore_warnings

sampling_methods = OrderedDict()
unary_operations = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt']
unary_operation_weights = np.asarray([5, 5, 5, 2, 2, 2, 4])
binary_operations = ['+', '-', '*', '/', '**']
binary_operation_weights = np.asarray([5, 5, 3, 3, 2])


def register_sampling_method(method):
    sampling_methods[method.__name__] = method


def generation_leaf(variables: list, constants: np.ndarray, constant_chance: float) -> Tuple:
    is_constant = np.random.random()

    if is_constant - constant_chance <= 0.01:
        random_constant = np.random.choice(constants)
        return True, ' {} '.format(str(random_constant))
    else:
        random_variable = np.random.choice(variables)
        return False, ' {} '.format(random_variable)


def generation_operator(variables: list, constants: np.ndarray, unary_chance: float, constant_chance: float) -> Tuple:
    is_unary = np.random.random()

    if is_unary - unary_chance <= 0.01:
        random_unary = np.random.choice(unary_operations, p=unary_operation_weights/unary_operation_weights.sum())
        random_binary = np.random.choice(binary_operations, p=binary_operation_weights/binary_operation_weights.sum())
        was_constant, equation_part = generation_leaf(variables, constants, constant_chance)

        equation_normal = ' ' + random_unary + '(' + equation_part + ') ' + random_binary
        equation_placeholder = random_unary + '( C ) ' + random_binary

        if was_constant:
            return equation_normal, equation_placeholder
        else:
            return equation_normal, equation_normal
    else:
        random_binary = np.random.choice(binary_operations)
        was_constant, equation_part = generation_leaf(variables, constants, constant_chance)

        equation_normal = equation_part + random_binary
        equation_placeholder = ' C ' + random_binary

        if was_constant:
            return equation_normal, equation_placeholder
        else:
            return equation_normal, equation_normal


def generate_equation_string(equation_length: int,
                             num_variables: int,
                             num_constants: int,
                             unary_chance: float,
                             constant_chance: float) -> str:
    constants = np.random.uniform(1, 5, num_constants)
    variables = ['x{}'.format(x) for x in range(num_variables)]
    equation_str = ''
    equation_str_constant_placeholder = ''
    was_unary = False

    for i, x in enumerate(range(equation_length)):
        if i == equation_length - 1:
            was_constant, equation_part = generation_leaf(variables, constants, constant_chance)
            if was_constant:
                equation_str_constant_placeholder += ' C '
            else:
                equation_str_constant_placeholder += equation_part
            equation_str += equation_part
        else:
            equation_part, equation_part_ph = generation_operator(variables, constants, unary_chance, constant_chance)
            equation_str += equation_part
            equation_str_constant_placeholder += equation_part_ph

    # check if at least one x is in equation
    if 'x' not in equation_str:
        random_variable = np.random.choice(variables)
        random_operation = np.random.choice(binary_operations, p=binary_operation_weights/binary_operation_weights.sum())
        equation_str += random_operation + random_variable
        equation_str_constant_placeholder += random_operation + random_variable

    return equation_str, equation_str_constant_placeholder


def generate_equation(iteration,
                      equation_length: int = 5,
                      num_variables: int = 10,
                      num_constants: int = 0.5,
                      unary_chance: float = 0.1,
                      constant_chance: float = 0.25) -> list:
    variables = ['x{}'.format(x) for x in range(num_variables)]
    equation, equation_placeholder = generate_equation_string(equation_length=equation_length,
                                                              num_variables=num_variables,
                                                              num_constants=num_constants,
                                                              unary_chance=unary_chance,
                                                              constant_chance=constant_chance)
    return equation, equation_placeholder


@register_sampling_method
def uniform_sampling(variable_count, sample_size, min_value, max_value):
    # sample number of positive values from uniform distribution
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives

    # convert to log values for uniform sampling
    log10_min_value = np.log10(np.clip(min_value, 0.1, 100.0))
    log10_max_value = np.log10(np.clip(max_value, 0.1, 100.0))

    # raise the samples to the power of 10 to remove log influence
    positive_samples = 10.0 ** np.random.uniform(log10_min_value, log10_max_value,
                                                 size=(variable_count, num_positives))
    negative_samples = -10.0 ** np.random.uniform(log10_min_value, log10_max_value,
                                                  size=(variable_count, num_negatives))
    all_samples = np.concatenate([positive_samples, negative_samples], axis=1)

    # np.random.shuffle(all_samples)
    all_samples = all_samples[:, np.random.permutation(all_samples.shape[1])]
    return all_samples


@register_sampling_method
def uniform_positive_sampling(variable_count, sample_size, min_value, max_value):
    # convert to log values for uniform sampling
    log10_min_value = np.log10(np.clip(min_value, 0.1, 100.0))
    log10_max_value = np.log10(np.clip(max_value, 0.1, 100.0))

    # raise the samples to the power of 10 to remove log influence
    positive_samples = 10.0 ** np.random.uniform(log10_min_value, log10_max_value, size=(variable_count, sample_size))

    # np.random.shuffle(positive_samples)
    positive_samples = positive_samples[:, np.random.permutation(positive_samples.shape[1])]
    return positive_samples


@register_sampling_method
def uniform_negative_sampling(variable_count, sample_size, min_value, max_value):
    # convert to log values for uniform sampling
    log10_min_value = np.log10(np.clip(min_value, 0.1, 100.0))
    log10_max_value = np.log10(np.clip(max_value, 0.1, 100.0))

    # raise the samples to the power of 10 to remove log influence
    negative_samples = -10.0 ** np.random.uniform(log10_min_value, log10_max_value, size=(variable_count, sample_size))

    # np.random.shuffle(negative_samples)
    negative_samples = negative_samples[:, np.random.permutation(negative_samples.shape[1])]
    return negative_samples


@register_sampling_method
def uniform_simple_sampling(variable_count, sample_size, min_value, max_value):
    # sample number of positive values from uniform distribution
    num_positives = sum(np.random.uniform(0.0001, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives

    # sample twice and concatenate
    positive_samples = np.random.uniform(0.0001, 1.0, size=(variable_count, num_positives))
    negative_samples = -np.random.uniform(0.0001, 1.0, size=(variable_count, num_negatives))
    all_samples = np.concatenate([positive_samples, negative_samples], axis=1)

    # np.random.shuffle(all_samples)
    all_samples = all_samples[:, np.random.permutation(all_samples.shape[1])]
    return all_samples


@register_sampling_method
def uniform_simple_positive_sampling(variable_count, sample_size, min_value, max_value):
    positive_samples = np.random.uniform(0.0001, 1.0, size=(variable_count, sample_size))
    # np.random.shuffle(positive_samples)
    positive_samples = positive_samples[:, np.random.permutation(positive_samples.shape[1])]
    return positive_samples


@register_sampling_method
def uniform_simple_negative_sampling(variable_count, sample_size, min_value, max_value):
    negative_samples = -np.random.uniform(0.0001, 1.0, size=(variable_count, sample_size))
    # np.random.shuffle(negative_samples)
    negative_samples = negative_samples[:, np.random.permutation(negative_samples.shape[1])]
    return negative_samples


@register_sampling_method
def integer_sampling(variable_count, sample_size, min_value, max_value):
    # sample number of positive values from uniform distribution
    num_positives = sum(np.random.uniform(0.0001, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives

    positive_samples = np.random.randint(min_value, max_value, size=(variable_count, num_positives))
    negative_samples = -np.random.randint(min_value, max_value, size=(variable_count, num_negatives))
    all_samples = np.concatenate([positive_samples, negative_samples], axis=1)

    # np.random.shuffle(all_samples)
    all_samples = all_samples[:, np.random.permutation(all_samples.shape[1])]
    return all_samples


@register_sampling_method
def integer_positive_sampling(variable_count, sample_size, min_value, max_value):
    positive_samples = np.random.randint(1, max_value, size=(variable_count, sample_size))
    # np.random.shuffle(all_samples)
    all_samples = positive_samples[:, np.random.permutation(positive_samples.shape[1])]
    return all_samples


@register_sampling_method
def integer_negative_sampling(variable_count, sample_size, min_value, max_value):
    negative_samples = np.random.randint(min_value, 1, size=(variable_count, sample_size))
    # np.random.shuffle(all_samples)
    all_samples = negative_samples[:, np.random.permutation(negative_samples.shape[1])]
    return all_samples


def sample_equation_with_sampler(equation, sympy_equation, variable_count,
                                 sample_size, min_value, max_value,
                                 sample_method, equation_min_max,
                                 sample_per_variable):
    equation_data = {}
    samples = None
    if sample_per_variable:
        samples_variables = []
        for variable in range(variable_count):
            if sample_method in sampling_methods.keys():
                samples = sampling_methods[sample_method](1, sample_size, min_value, max_value)
                samples_variables.append(samples)
            else:
                # sample_method == 'random
                random_sampler = np.random.choice(list(sampling_methods.keys()))
                samples = sampling_methods[random_sampler](1, sample_size, min_value, max_value)
                samples_variables.append(samples)
        samples = np.squeeze(np.asarray(samples_variables))
    else:
        if sample_method in sampling_methods.keys():
            samples = sampling_methods[sample_method](variable_count, sample_size, min_value, max_value)
        else:
            # sample_method == 'random
            random_sampler = np.random.choice(list(sampling_methods.keys()))
            samples = sampling_methods[random_sampler](variable_count, sample_size, min_value, max_value)

    try:
        with ignore_warnings(RuntimeWarning):
            sample_values = sympy_equation(*samples)
    except ValueError as ve:
        # print('Encountered NaN or negative power values.')
        return None
    except RuntimeWarning:
        # print('Power was too high to calculate.')
        return None
    except TypeError:
        # complex numbers
        return None
    except OverflowError:
        # too big integer numbers
        return None
    except Exception as e:
        warnings.warn(str(e))
        # print("Generic error occurred")
        return None

    if np.isscalar(sample_values):
        # print('Sample values were a scalar.')
        return None

    if any(np.isnan(sample_values)):
        # print('NaN value in sampling results.')
        return None

    if any(np.isinf(sample_values)):
        # print('Inf value in sampling results.')
        return None

    if sample_values.max() >= equation_min_max or sample_values.min() <= -equation_min_max:
        # values were lower or greater than the set bounds
        return None

    if sample_values.min() == sample_values.max():
        # equation was constant
        return None

    equation_data['string'] = equation
    # equation_data['lambda'] = sympy_equation
    equation_data['x_samples'] = samples.tolist()
    equation_data['y_samples'] = sample_values.tolist()

    return equation_data


def sample_equation(args, variable_count: int, sample_size: int, min_value, max_value, sample_method, equation_min_max, sample_per_variable):
    # equation_str, lambda_equation = args
    equation_str = args
    sympy_equation = sympy.parse_expr(equation_str)
    lambda_equation = sympy.lambdify(['x{}'.format(x) for x in range(variable_count)], sympy_equation, modules='numpy')
    equation_data = sample_equation_with_sampler(equation_str,
                                                 lambda_equation,
                                                 variable_count,
                                                 sample_size,
                                                 min_value,
                                                 max_value,
                                                 sample_method,
                                                 equation_min_max,
                                                 sample_per_variable)
    if equation_data is not None:
        return equation_data
    else:
        return None


def do_pool_task(pool_size, function_to_pool, tasks, timeout, sample_time, kwargs_dict):
    pool = multiprocessing.Pool(processes=pool_size)

    processes = []
    results = []
    # Submit tasks to the pool
    for task in tasks:
        result = pool.apply_async(partial(function_to_pool, **kwargs_dict), (task,))
        processes.append(result)

    # Wait for the results with a timeout
    start_time = time.time()
    all_completed = False
    while not all_completed and time.time() - start_time < timeout:
        all_completed = all([result.ready() for result in processes])
        completed_processes = [process for process in processes if process.ready()]

        if completed_processes:
            # Gather completed results
            for process in completed_processes:
                try:
                    process.wait(timeout=timeout)  # Ensure result retrieval
                    results.append(process.get(timeout=timeout))
                except multiprocessing.TimeoutError:
                    warnings.warn('During While: Pool item timed out!')
                    pool.close()
                    pool.join()
                    return results
                processes.remove(process)

        time.sleep(sample_time)  # Wait for some time before checking again

    if not all_completed:
        warnings.warn('After while: Ending without all threads completed!')

    pool.close()
    pool.join()

    return results


def do_pool_task_one_pool(pool, function_to_pool, tasks, timeout, sample_time, kwargs_dict):
    processes = []
    results = []
    # Submit tasks to the pool
    for task in tasks:
        result = pool.apply_async(partial(function_to_pool, **kwargs_dict), (task,))
        processes.append(result)

    # Wait for the results with a timeout
    start_time = time.time()
    all_completed = False
    while not all_completed and time.time() - start_time < timeout:
        all_completed = all([result.ready() for result in processes])
        completed_processes = [process for process in processes if process.ready()]

        if completed_processes:
            # Gather completed results
            for process in completed_processes:
                try:
                    process.wait(timeout=timeout)  # Ensure result retrieval
                    results.append(process.get(timeout=timeout))
                except multiprocessing.TimeoutError:
                    continue
                processes.remove(process)

        time.sleep(sample_time)  # Wait for some time before checking again

    return results


def generate_equation_dataset_pool_one_timeout(num_equation_sets: int,
                                               equation_length=5,
                                               num_variables=10,
                                               num_constants=5,
                                               unary_chance=0.1,
                                               constant_chance=0.25,
                                               sample_size=50,
                                               min_value=-20,
                                               max_value=20,
                                               pool_size=5,
                                               chunk_size=1000,
                                               timeout=120,
                                               sample_time=0.1,
                                               equation_min_max=100,
                                               sample_per_variable=False):
    datasets = []
    equations_generated = 0
    # while equations_generated < num_equation_sets:
    kwargs_generator = {'equation_length': equation_length,
                        'num_variables': num_variables,
                        'num_constants': num_constants,
                        'unary_chance': unary_chance,
                        'constant_chance': constant_chance}

    kwargs_sampler = {'variable_count': num_variables,
                      'sample_size': sample_size,
                      'min_value': min_value,
                      'max_value': max_value,
                      'equation_min_max': equation_min_max,
                      'sample_per_variable': sample_per_variable}

    temp_file = './equation_samples/temp___{}_{}.pickle'.format(num_equation_sets, time.time_ns())
    temp_log_file = './equation_samples/temp_log___{}_{}.txt'.format(num_equation_sets, time.time_ns())

    tasks = range(chunk_size)
    results = []

    pool = multiprocessing.Pool(processes=pool_size)

    while equations_generated < num_equation_sets:

        print("Generating step... {}/{}".format(equations_generated, num_equation_sets))

        equations = do_pool_task_one_pool(pool, generate_equation, tasks, timeout, sample_time, kwargs_generator)
        equations_str_list = [x[0] for x in equations]

        samples = do_pool_task_one_pool(pool, sample_equation, equations_str_list, timeout, sample_time, kwargs_sampler)

        if samples is None:
            continue

        not_none_index = [samples.index(x) for x in samples if x is not None]
        for index in not_none_index:
            if equations_generated == num_equation_sets:
                pool.close()
                pool.join()
                return datasets

            # samples[index]['placeholder'] = equations[index][2]
            samples[index]['placeholder'] = equations[index][1]
            datasets.append(samples[index])
            equations_generated += 1

        pickle.dump(datasets, open(temp_file, 'wb+'))  # overwrite with datasets or append fresh_generated?
        with open(temp_log_file, 'a+') as file:
            file.write(str(equations_generated) + '\n')

    pool.close()
    pool.join()
    return datasets


def generate_equation_dataset_pool_timeout(num_equation_sets: int,
                                           equation_length=5,
                                           num_variables=10,
                                           num_constants=5,
                                           unary_chance=0.1,
                                           constant_chance=0.25,
                                           sample_size=50,
                                           min_value=-20,
                                           max_value=20,
                                           pool_size=5,
                                           chunk_size=1000,
                                           timeout=120,
                                           sample_time=0.1,
                                           sample_method='random',
                                           equation_min_max=100,
                                           sample_per_variable=False):
    datasets = []
    equations_generated = 0

    # while equations_generated < num_equation_sets:
    kwargs_generator = {'equation_length': equation_length,
                        'num_variables': num_variables,
                        'num_constants': num_constants,
                        'unary_chance': unary_chance,
                        'constant_chance': constant_chance}

    kwargs_sampler = {'variable_count': num_variables,
                      'sample_size': sample_size,
                      'min_value': min_value,
                      'max_value': max_value,
                      'sample_method': sample_method,
                      'equation_min_max': equation_min_max,
                      'sample_per_variable': sample_per_variable}

    temp_file = './equation_samples/temp___{}_{}.pickle'.format(num_equation_sets, time.time_ns())
    temp_log_file = './equation_samples/temp_log___{}_{}.txt'.format(num_equation_sets, time.time_ns())

    tasks = range(chunk_size)
    results = []

    while equations_generated < num_equation_sets:

        print("Generating step... {}/{}".format(equations_generated, num_equation_sets))

        print("Generating equations...")
        equations = do_pool_task(pool_size, generate_equation, tasks, timeout, sample_time, kwargs_generator)
        equations_str_list = [x[0] for x in equations]

        print("Sampling equations...")
        samples = do_pool_task(pool_size, sample_equation, equations_str_list, timeout, sample_time, kwargs_sampler)

        if samples is None:
            continue

        not_none_index = [samples.index(x) for x in samples if x is not None]
        for index in not_none_index:
            if equations_generated == num_equation_sets:
                return datasets

            samples[index]['placeholder'] = equations[index][1]
            datasets.append(samples[index])
            equations_generated += 1

    return datasets


def generate_equation_dataset_pool_map(num_equation_sets: int,
                                       equation_length=5,
                                       num_variables=10,
                                       num_constants=5,
                                       unary_chance=0.1,
                                       constant_chance=0.25,
                                       sample_size=50,
                                       min_value=-20,
                                       max_value=20,
                                       pool_size=5,
                                       chunk_size=1000):
    datasets = []
    equations_generated = 0

    # while equations_generated < num_equation_sets:
    kwargs_generator = {'equation_length': equation_length,
                        'num_variables': num_variables,
                        'num_constants': num_constants,
                        'unary_chance': unary_chance,
                        'constant_chance': constant_chance}

    kwargs_sampler = {'variable_count': num_variables,
                      'sample_size': sample_size,
                      'min_value': min_value,
                      'max_value': max_value}

    temp_file = './equation_samples/temp___{}_{}.pickle'.format(num_equation_sets, time.time_ns())
    temp_log_file = './equation_samples/temp_log___{}_{}.txt'.format(num_equation_sets, time.time_ns())

    pool = multiprocessing.Pool(processes=pool_size)
    tasks = range(chunk_size)
    results = []
    timeout = 500

    while equations_generated < num_equation_sets:

        print("Generating step... {}/{}".format(equations_generated, num_equation_sets))

        equations = None
        try:
            equations = pool.map(partial(generate_equation, **kwargs_generator), range(chunk_size))
        except Exception as exception:
            print("Generation: Something went wrong. {}".format(str(exception)))
        # equations = list(equations)

        if equations is None or len(equations) == 0:
            continue

        # equation_lambda_pair = list(zip([x[0] for x in equations], [x[1] for x in equations]))
        equations_str_list = [x[0] for x in equations]

        samples = None
        try:
            # samples = pool.map(partial(sample_equation, **kwargs_sampler), equation_lambda_pair)
            samples = pool.map(partial(sample_equation, **kwargs_sampler), equations_str_list)
        except Exception as exception:
            print("Sampling: Something went wrong.".format(str(exception)))
        # samples = list(samples)

        if samples is None:
            continue

        not_none_index = [samples.index(x) for x in samples if x is not None]

        for index in not_none_index:
            if equations_generated == num_equation_sets:
                pool.close()
                pool.join()
                return datasets

            # samples[index]['placeholder'] = equations[index][2]
            samples[index]['placeholder'] = equations[index][1]
            datasets.append(samples[index])
            equations_generated += 1

        pickle.dump(datasets, open(temp_file, 'wb+'))  # overwrite with datasets or append fresh_generated?
        with open(temp_log_file, 'a+') as file:
            file.write(str(equations_generated) + '\n')

    pool.close()
    pool.join()
    return datasets


def generate_equation_dataset(num_equation_sets: int,
                              equation_length=5,
                              num_variables=10,
                              num_constants=5,
                              unary_chance=0.1,
                              constant_chance=0.25,
                              sample_size=50,
                              min_value=-20,
                              max_value=20,
                              pool_size=5,
                              chunk_size=1000):
    datasets = []
    equations_generated = 0

    while equations_generated < num_equation_sets:
        equation_str, lambda_equation, equation_placeholder = generate_equation(0,
                                                                                equation_length=equation_length,
                                                                                num_variables=num_variables,
                                                                                num_constants=num_constants,
                                                                                unary_chance=unary_chance,
                                                                                constant_chance=constant_chance)

        equations_sampled = sample_equation((equation_str,
                                            lambda_equation),
                                            variable_count=num_variables,
                                            sample_size=sample_size,
                                            min_value=min_value,
                                            max_value=max_value)

        if equations_sampled is not None:
            equations_sampled['placeholder'] = equation_placeholder
            equations_generated += 1
            datasets.append(equations_sampled)
    return datasets


def generate_dataset(dataset_size, **kwargs):
    try:
        # dataset = generate_equation_dataset(dataset_size, **kwargs)
        # dataset = generate_equation_dataset_pool_map(dataset_size, **kwargs)
        dataset = generate_equation_dataset_pool_timeout(dataset_size, **kwargs)
        # dataset = generate_equation_dataset_pool_one_timeout(dataset_size, **kwargs)
    except Exception as exception:
        print('Saving: Something went wrong. {}'.format(str(exception)))
        raise exception

    return dataset


if __name__ == '__main__':

    equation_data = generate_equation(0, 5, 10, 20, 0.2, 0.3)
    sample_data = sample_equation(equation_data[0], 10, 1000, -30, 30, 'random', 100, True)

