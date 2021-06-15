import numpy as np


def minibatches(data, batch_size):
    # d:(h, t, label)
    # d[0]:features vector (36,)
    # len(data) 条训练数据，记为 n
    # x:(n,36) y:(n,) one_hot:(n,3)
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    # [x, one_hot]:[ndarray (n,36), ndarray (n,3)]
    # @return : generator
    # 其中 generator 的元素：[ndarray (batch_size,36), ndarray (batch_size,3)]
    return get_minibatches([x, one_hot], batch_size)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    # True:a list where each element is either a list or numpy array
    # False: list or numpy array or others type that would cause exceptions
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    # [0 1 2 3 4 ... data_size] ndarray (data_size,)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    # [0, minibatch_size, 2 * minibatch_size,...] ndarray
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        # ceil(data_size/minibatch_size)次循环,记为 n
        # 将 indices 划分为 n 块,每块的长度是 minibatch_size(最后一块除外)
        # 第一个 minibatch_indices 可能的情况：[0,1,...,minibatch_start -1]
        # 或 [3,17,132,...] 因为 np.random.shuffle(indices)
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        # [ndarray,...] or [list,...] or ndarray or list
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    # data : ndarray or list
    # 若 data : (N,*) minibatch_idx : (L,) ,则 data[minibatch_idx] (L,*)
    # [data[i] for i in minibatch_idx] 长度为 L 的 list
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
