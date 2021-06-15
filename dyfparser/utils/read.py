def read_conllu(filename: str, lowercase=False, max_sentence: int = None):
    sentences = []
    with open(filename, encoding='utf=8') as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                # ['41', 'fiscal', '_', 'ADJ', 'JJ', '_', '42', 'amod', '_', '_']
                word.append(sp[1].lower() if lowercase else sp[1])
                pos.append(sp[4])
                head.append(int(sp[6]))
                label.append(sp[7])
            elif len(word) > 0:
                # ['']
                sentences.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_sentence is not None) and (len(sentences) == max_sentence):
                    break
        # 收尾,因为最后一行不一定是['']
        if len(word) > 0:
            sentences.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return sentences


def read_wordvec(wordvec_path):
    word_vectors = {}
    # 130000 行的 list
    for line in open(wordvec_path).readlines():
        # 去掉\n  51维[单词，50维词向量]
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    return word_vectors


def read_doc(doc):
    dataset = []
    for sent in doc.sents:
        word, pos = [], []
        for token in sent:
            word.append(token.text)
            pos.append(token.tag_)
        n = len(word)
        dataset.append({'word': word, 'pos': pos,
                        'head': [-1] * n, 'label': [''] * n})
    return dataset
