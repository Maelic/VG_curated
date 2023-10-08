import numpy as np

def imbalance_ratio(data,category):
    value_counts = data[category].value_counts()
    counts = [count for category, count in value_counts.items()]
    count_max = max(counts)
    count_min = min(counts)
    IR = count_max / count_min
    return IR

def hellinger_distance(vec1, vec2):
    return np.sqrt(np.sum((np.sqrt(vec1) - np.sqrt(vec2)) ** 2)) / np.sqrt(2)

def imbalance_degree(data, category, c):
    value_counts = data[category].value_counts()
    counts = [count for category, count in value_counts.items()]
    size = len(data)
    p = np.array(counts) / size
    if (len(p) < c):
        p = np.concatenate((p, [0] * (c-len(p))))
    b = [1/c] * c
    m = 0
    for prob in (p-b):
        if prob < 0:
            m +=1
    p_m = [0] * m + [1/c] * (c-m-1) + [1-(c-m-1)/c]
    ID = (hellinger_distance(p,b) / hellinger_distance(p_m,b)) + (m-1)
    return ID

def log_likelihood_index(data, category):
    c = len(np.unique(data[category]))
    value_counts = data[category].value_counts()
    size = len(data)
    proba = [(count/size) for category, count in value_counts.items()]
    summation = np.sum([(p*np.log(p*c)) for p in proba])
    LLI = 2 * summation
    return LLI