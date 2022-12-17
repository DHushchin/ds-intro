import numpy as np
import pandas as pd


def voronine(data, criteria):
    n = len(data.iloc[:, 0])
    k = len(data.iloc[0, :])
    F_norm = np.zeros((k, n))
    G = np.ones((k,)) / k
    F_sum = np.zeros((k,))
    
    for i in range(k):
        if criteria[data.keys()[i]][0] == 'max':
            F_sum[i] = (1 / data.values[:, i]).sum()
        else:
            F_sum[i] = data.values[:, i].sum()
            
    integro = np.zeros((n,))
    
    for i in range(k):
        if criteria[data.keys()[i]][0] == 'max':
            F_norm[i] = 1 / data.values[:, i] / F_sum[i]
        else:
            F_norm[i] = data.values[:, i] / F_sum[i]
            
    for i in range(n):
        integro[i] = np.array([G[j] * (1 - F_norm[j, i]) ** -1 for j in range(k)]).sum()
        
        
    return np.argmin(integro) 


def main():
    data = pd.read_csv('data.csv')
    criteria = pd.read_csv('criteria.csv')
    print(f'Optimal variant: {voronine(data, criteria)}')
    print(data.iloc[voronine(data, criteria), :])


if __name__ == '__main__':
    main()
