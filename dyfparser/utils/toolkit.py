from collections import Counter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.__count = 0
        self.__sum = 0
        self.__lastval = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f'n must be more than 0,n:{n}')
        self.__lastval = val
        self.__sum += val * n
        self.__count += n

    @property
    def sum(self):
        return self.__sum

    @property
    def count(self):
        return self.__count

    @property
    def avg(self):
        return self.__sum / self.__count


def isbool_or_exception(value):
    if not isinstance(value, bool):
        raise ValueError(f'value must be True or False,value: {value}')
    # isinstance(value, bool) == True
    return True


def is_only_root(dataset):
    root_labels = list([l for ex in dataset
                        for (h, l) in zip(ex['head'], ex['label']) if h == 0])
    counter = Counter(root_labels)
    return len(counter) == 1


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def dict_reversemap(srcdict):
    return {v: k for (k, v) in srcdict.items()}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)
