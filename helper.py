import numpy as np
import pandas as pd

def make_embeddings(monthly_data, size=60):
    ids = np.unique(monthly_data.session_id.to_list())
    embeddings = np.zeros((len(ids), size))
    for i, id_ in enumerate(ids):
        prev_min = 0
        data = monthly_data[monthly_data.session_id == id_]
        for time, price in zip(data.time.to_list(), data.price.to_list()):
            if prev_min == 0 and i > 0:
                embeddings[i, :] = embeddings[i-1, -1]
            minute = int(time.split(':')[1])
            if prev_min > minute:
                continue
            embeddings[i, minute + 1:] = price
            prev_min = minute
        embeddings[i, prev_min:] = embeddings[i, prev_min]
    return embeddings


def pretty_print(mtx):
    print('-', end='\t')
    for i in range(mtx.shape[1]):
        print(i, end='\t')
    print()
    for i, row in enumerate(mtx):
        print(i, end='\t')
        for j, col in enumerate(row):
            print(round(mtx[i, j], 2), end='\t')
        print()

        
def normalise(mtx, eps=1e-8):
    output = np.zeros(mtx.shape)
    for i, row in enumerate(mtx):
        maxx = row.max()
        minn = row.min()
        output[i, :] = (row - minn) / (maxx - minn + eps)
    return output