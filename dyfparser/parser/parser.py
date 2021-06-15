from dyfparser.parser.parserconfig import ParserConfig
from dyfparser.parser.parserconstant import ParserConstant
from dyfparser.utils.read import read_conllu, read_wordvec
from dyfparser.utils.toolkit import *
from dyfparser.model.parsermodel import ParserModel
from dyfparser.model.modelwrapper import ModelWrapper
from dyfparser.parser.parseruntime import minibatch_parse
from dyfparser.filepath import FilePath

import numpy as np
from tqdm import tqdm
from collections import Counter
import json
import torch


class Parser(object):

    def __init__(self, conllu_path: str = None, wordvec_path: str = None, max_sen: int = None,
                 config: ParserConfig = None):
        if conllu_path is None or wordvec_path is None:
            return
        self.config = config if isinstance(config, ParserConfig) else ParserConfig.defaultConfig()
        self.dataset = read_conllu(conllu_path, self.config.lowercase, max_sen)
        if not is_only_root(self.dataset):
            raise ValueError('Warning: more than one root label')
        # 48,36,30,18
        self.n_features = 18 + (18 if self.config.use_pos else 0) + (12 if self.config.use_dep else 0)
        # ['root', 'csubjpass', 'dep',...]
        # 第一个元素是root,与excel第16行的属性一致
        deprel = [ParserConstant.root_label] + list(set([w for ex in self.dataset
                                                         for w in ex['label']
                                                         if w != ParserConstant.root_label]))
        self.n_deprel = 1 if self.config.unlabeled else len(deprel)
        self.shortcuts = {}
        self.tok2id = self.__gentok2id(deprel, self.dataset)
        self.id2tok = dict_reversemap(self.tok2id)
        self.n_tokens = len(self.tok2id)
        self.tran2id = self.__gentran2id(deprel)
        self.id2tran = dict_reversemap(self.tran2id)
        self.n_trans = len(self.tran2id)
        self.model = ParserModel(self.__genembeddings(wordvec_path))

    def __gentok2id(self, deprel, dataset):
        # {'<l>:root': 0, '<l>:mwe': 1, '<l>:appos': 2, ...}
        # <l>excel第16行的属性：索引
        tok2id = {ParserConstant.L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        # '<l>:<NULL>': 39
        # self.L_NULL = 39 <l>:NULL的索引，单独拿出来方便使用
        tok2id[ParserConstant.L_PREFIX + ParserConstant.NULL] = self.shortcuts['L_NULL'] = len(tok2id)
        # Build dictionary for part-of-speech tags
        # {'<p>:NN': 40, '<p>:IN': 41, '<p>:NNP': 42, '<p>:DT': 43,...}
        # build_dict 返回上一行的字典，已去重，取频度前 n_max 的元素
        tok2id.update(build_dict([ParserConstant.P_PREFIX + w for ex in dataset for w in ex['pos']],
                                 offset=len(tok2id)))
        #  '<p>:<UNK>': 84, '<p>:<NULL>': 85, '<p>:<ROOT>': 86
        tok2id[ParserConstant.P_PREFIX + ParserConstant.UNK] = self.shortcuts['P_UNK'] = len(tok2id)
        tok2id[ParserConstant.P_PREFIX + ParserConstant.NULL] = self.shortcuts['P_NULL'] = len(tok2id)
        tok2id[ParserConstant.P_PREFIX + ParserConstant.ROOT] = self.shortcuts['P_ROOT'] = len(tok2id)

        # Build dictionary for words.
        # {...,'some': 178, 'net': 179, 'index': 180, ...}
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                 offset=len(tok2id)))
        # '<UNK>': 6927, '<NULL>': 6928, '<ROOT>': 6929
        tok2id[ParserConstant.UNK] = self.shortcuts['UNK'] = len(tok2id)
        tok2id[ParserConstant.NULL] = self.shortcuts['NULL'] = len(tok2id)
        tok2id[ParserConstant.ROOT] = self.shortcuts['ROOT'] = len(tok2id)
        return tok2id

    def __gentran2id(self, deprel):
        trans = ['L', 'R', 'S'] if self.config.unlabeled \
            else ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
        # {'L-root': 0, 'L-discourse': 1, ...}
        return {t: i for (i, t) in enumerate(trans)}

    def __genembeddings(self, wordvec_path):
        word_vectors = read_wordvec(wordvec_path)
        # numpy.asarray(a, dtype=None, order=None, *, like=None)
        #  numpy.random.normal(loc=0.0, scale=1.0, size=None)
        # parser.n_tokens行, 50列的满足高斯分布的随机数矩阵
        embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (self.n_tokens, 50)), dtype='float32')

        for token in self.tok2id:
            i = self.tok2id[token]
            # 在 word_vectors 查已经有的词向量，并初始化
            if token in word_vectors:
                embeddings_matrix[i] = word_vectors[token]
            elif token.lower() in word_vectors:
                embeddings_matrix[i] = word_vectors[token.lower()]
            # 在word_vectors 查不到的使用高斯分布随机初始化
            return embeddings_matrix

    def vectorize(self, sentences):
        # len(sentences) == len(vec_sentences)
        # 每句话前加 ROOT , self.tok2id 中的找不到的单词变成 UNK
        # ROOT,UNK,每句话中的单词全部变成索引（数字）
        vec_sentences = []
        for ex in sentences:
            # self.ROOT,self.UNK,单词在tok2id中的索引
            word = [self.shortcuts['ROOT']] + [self.tok2id[w] if w in self.tok2id
                                               else self.shortcuts['UNK'] for w in ex['word']]
            # self.P_ROOT,self.P_UNK,<p>:pos在tok2id中的索引
            pos = [self.shortcuts['P_ROOT']] + [
                self.tok2id[ParserConstant.P_PREFIX + w] if ParserConstant.P_PREFIX + w in self.tok2id
                else self.shortcuts['P_UNK'] for w in ex['pos']]
            head = [-1] + ex['head']
            # -1,-1,<l>:lable在tok2id中的索引
            label = [-1] + [self.tok2id[ParserConstant.L_PREFIX + w] if ParserConstant.L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_sentences.append({'word': word, 'pos': pos,
                                  'head': head, 'label': label})
        return vec_sentences

    def create_instances(self, sentences):
        all_instances = []
        succ = 0
        # sentences：数据集      句子数：len(sentences)
        # ex：一句话            每个 ex 都执行一次 for...else 结构
        # 遍历结束后：共执行 len(sentences) 次 for...else 结构
        # 其中执行了 succ 次 else 结构，len(sentences) - succ 次 break 语句
        for ex in sentences:
            # 一句话中 真 单词数(不算ROOT)
            n_words = len(ex['word']) - 1

            instances = []

            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []

            # for ... else 结构
            # n_words * 2 次循环，每次添加一条训练数据
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                # 成功添加一条训练数据：(h, t, label)
                # h:长度为 36 的 features list
                # t:长度为 3
                # label:标量
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                # 更新stack,buf,arcs
                if gold_t == self.n_trans - 1:
                    # shift
                    stack.append(buf[0])
                    buf.pop(0)
                elif gold_t < self.n_deprel:
                    # left arcs
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack.pop(-2)
                else:
                    # right arcs
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack.pop()
            else:
                succ += 1
                # len(instances) == 2 * n_words
                all_instances += instances
        # len(all_instances) == sum(len(instances))
        # 即对于所有执行过 succ+=1 的句子 sum(2 * n_words)
        # all_instances 的元素：三元组 (h, t, label)
        return all_instances

    def extract_features(self, stack, buf, arcs, ex):
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(lk):
            return sorted([arc[1] for arc in arcs if arc[0] == lk and arc[1] < lk])

        def get_rc(rk):
            return sorted([arc[1] for arc in arcs if arc[0] == rk and arc[1] > rk],
                          reverse=True)

        p_features = []
        l_features = []
        features = [self.shortcuts['NULL']] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        features += [ex['word'][x] for x in buf[:3]] + [self.shortcuts['NULL']] * (3 - len(buf))
        if self.config.use_pos:
            p_features = [self.shortcuts['P_NULL']] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.shortcuts['P_NULL']] * (3 - len(buf))

        for i in range(2):
            if i < len(stack):
                k = stack[-i - 1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.shortcuts['NULL'])
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.shortcuts['NULL'])
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.shortcuts['NULL'])
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.shortcuts['NULL'])
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.shortcuts['NULL'])
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.shortcuts['NULL'])

                if self.config.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.shortcuts['P_NULL'])
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.shortcuts['P_NULL'])
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.shortcuts['P_NULL'])
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.shortcuts['P_NULL'])
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.shortcuts['P_NULL'])
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.shortcuts['P_NULL'])

                if self.config.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.shortcuts['L_NULL'])
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.shortcuts['L_NULL'])
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.shortcuts['L_NULL'])
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.shortcuts['L_NULL'])
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.shortcuts['L_NULL'])
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.shortcuts['L_NULL'])
            else:
                features += [self.shortcuts['NULL']] * 6
                if self.config.use_pos:
                    p_features += [self.shortcuts['P_NULL']] * 6
                if self.config.use_dep:
                    l_features += [self.shortcuts['L_NULL']] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:
            return self.n_trans - 1

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.config.unlabeled:
            if (i1 > 0) and (h1 == i0):
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                    (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                    (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        vectorizedataset = self.vectorize(dataset)
        dependencies = self.__parse(vectorizedataset, eval_batch_size)
        UAS = self.__eval(vectorizedataset, dependencies)
        return UAS, dependencies

    def __eval(self, vectorizedataset, dependencies):
        UAS = all_tokens = 0.0
        with tqdm(total=len(vectorizedataset)) as prog:
            for i, ex in enumerate(vectorizedataset):
                head = [-1] * len(ex['word'])
                for h, t, in dependencies[i]:
                    head[t] = h
                for j, pred_h, gold_h, gold_l, pos in \
                        zip(range(1, len(ex['word'])), head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(ParserConstant.P_PREFIX)
                    pos_str = self.id2tok[pos][len(ParserConstant.P_PREFIX):]
                    if self.config.with_punct or (not punct(self.config.language, pos_str)):
                        UAS += 1 if pred_h == gold_h else 0
                        all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS

    def __parse(self, vectorizedataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(vectorizedataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, vectorizedataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)
        return dependencies

    def parsing(self, dataset):
        vecdata = self.vectorize(dataset)
        vecdep = self.__parse(vecdata)
        n = len(dataset)
        assert n == len(vecdep)
        deps = [
            {
                'words': [
                    {
                        'text': text,
                        'tag': tag
                    } for text, tag in zip(['ROOT'] + dataset[i]['word'], [''] + dataset[i]['pos'])
                ],
                'arcs': [
                    {'start': start,
                     'end': end,
                     'label': '',
                     'dir': 'right'
                     } for start, end in vecdep[i]
                ]
            } for i in range(n)
        ]
        return deps

    def get_trainset(self):
        return self.create_instances(self.vectorize(self.dataset)) if self.dataset else None

    def save(self, parser_path: str, weight_path: str) -> None:
        """

        :param parser_path: 句法分词器在磁盘中的位置
        :param weight_path: 神经网络模型的权重矩阵在磁盘中的位置
        :return:
        """
        serialize = [self.n_features, self.n_deprel, self.n_tokens, self.n_trans,
                     self.config.tolist(), self.shortcuts, self.tok2id, self.tran2id, weight_path]
        torch.save(self.model, weight_path)
        with open(parser_path, mode='w', encoding='utf-8') as f:
            json.dump(serialize, f)

    @classmethod
    def load(cls, parser_path):
        parser = Parser()
        parser.dataset = None
        with open(parser_path, encoding='utf-8') as f:
            [parser.n_features, parser.n_deprel,
             parser.n_tokens, parser.n_trans,
             parameters, parser.shortcuts,
             parser.tok2id, parser.tran2id,
             weight_path] = json.load(f)
        parser.id2tok = dict_reversemap(parser.tok2id)
        parser.id2tran = dict_reversemap(parser.tran2id)
        parser.config = ParserConfig(*parameters)
        parser.model = torch.load(weight_path)
        parser.model.eval()
        return parser

    @classmethod
    def default_parser(cls):
        return cls.load(FilePath.default_parser_file)
