import numpy as np
import matplotlib.pyplot as plt

SIZE = 10000
STD = 3


def error_model(size, mean, std):
    return np.random.normal(mean, std, size)


def monte_carlo(size, a, b):
    return np.random.rand(size) * (b - a) + a


def change_model(data):
    return data ** 2 / 2


def experimental_data(data, error):
    return np.add(data, error)


def generate_data(size):
    error = error_model(size, mean=0, std=STD)
    rand_data = monte_carlo(size, -10, 10)
    rand_data = np.sort(rand_data)
    trend_data = change_model(rand_data)
    experiment_data = experimental_data(trend_data, error)
    return rand_data, trend_data, experiment_data


def MNK(x, y):
    yin = np.reshape(y, (-1, 1))
    F = np.ones((len(x), 3))
    F[::, 1] = x
    F[::, 2] = x ** 2
    FT = F.T
    FT_F = FT.dot(F)
    FT_F_inv = np.linalg.inv(FT_F)
    FT_F_inv_FT = FT_F_inv.dot(FT)
    C = FT_F_inv_FT.dot(yin)
    yout = F.dot(C)
    return yout


def remove_outliers(x, y):
    data = np.array(list(zip(x, y)))
    window_size = 200
    
    i = 0
    while i < len(data) - window_size:
        mnk = MNK(data[i:i + window_size, 0], data[i:i + window_size, 1])
        std = np.std(data[i:i + window_size, 1])

        if abs(data[i + window_size - 1, 1] - mnk[-1]) > 3 * std:
            data = np.delete(data, i + window_size - 1, axis=0)
        else:
            i += 1
            
    return data[:, 0], data[:, 1]

def stats_info(data):
    print('Mean: ', np.mean(data))
    print('Median: ', np.median(data))
    print('Variance: ', np.var(data))
    print('Std: ', np.std(data), '\n')
    
    
def main():
    # Data processing
    rand_data, trend_data, experiment_data = generate_data(SIZE)
    yout = MNK(rand_data, experiment_data)
    rand_data_2, experiment_data_2 = remove_outliers(rand_data, experiment_data)  
    yout_2 = MNK(rand_data_2, experiment_data_2)
    
    # Data statistics
    print('Random data:')
    stats_info(rand_data)
    
    print('Trend data stats with outliers:')
    stats_info(trend_data)

    print('Experiment data stats with outliers:')
    stats_info(experiment_data)

    print('Random data after removing outliers stats:')
    stats_info(rand_data_2)

    print('Experiment data after removing outliers stats:')
    stats_info(experiment_data_2)

    print('Smoothed data stats:')
    stats_info(yout)

    # Data visualization
    plt.plot(experiment_data, 'b', label='with outliers')
    plt.plot(experiment_data_2, 'y', label='without outliers')
    plt.plot(trend_data, 'r', label='trend')
    plt.plot(yout_2, 'g', label='smoothed')
    plt.legend()
    plt.show()
    
    plt.hist(abs(experiment_data - trend_data), bins=20, facecolor='red', alpha=0.5, label='with outliers')
    plt.hist(abs(experiment_data_2 - change_model(rand_data_2)), bins=20, facecolor='blue', alpha=0.5, label='without outliers')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()
    