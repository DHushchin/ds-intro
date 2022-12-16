import numpy as np
import matplotlib.pyplot as plt

SIZE = 10000

def error_model(size, distribution, mean=0, std=1):
    if distribution == 'normal':
        return np.random.normal(mean, std, size)
    elif distribution == 'exponential':
        return np.random.exponential(std, size) * 2
    else:
        raise ValueError('Unknown distribution')

def change_model(change_method):
    if change_method == 'linear':
        return np.array([0.01 * i for i in range(SIZE)])
    elif change_method == 'quadratic':
        return np.array([0.000003 * i**2 for i in range(SIZE)])
    else:
        raise ValueError('Unknown change method')

def experimental_data(data, error):
    return np.add(data, error)

def generate_data(size, error_distribution, change_distribution):
    error = error_model(size, error_distribution)
    data = change_model(change_distribution)
    experiment_data = experimental_data(data, error)
    return data, experiment_data, error

def stats_info(data):
    titles = ['Data', 'Experimental data', 'Error']
    for i in range(len(data)):
        print(titles[i])
        print('Mean: ', np.mean(data[i]))
        print('Median: ', np.median(data[i]))
        print('Variance: ', np.var(data[i]))
        print('Std: ', np.std(data[i]), '\n')

def plot_data(data, exp_data, error):
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot(data, color='red')
    axes[0].plot(exp_data, alpha=0.6, color='yellow')
    axes[0].set_title('Data and experimental data')
    axes[1].hist(error, bins=20)
    axes[1].set_title('Error')
    plt.show()

def info(error, change):
    print(f'Error distribution: {error}')
    print(f'Change distribution: {change}\n')
    data, exp_data, error = generate_data(SIZE, error, change)
    plot_data(data, exp_data, error)
    stats_info((data, exp_data, error))

info('normal', 'linear')
info('normal', 'quadratic')
info('exponential', 'linear')
info('exponential', 'quadratic')
