from dyfparser.parser.parser import Parser
from dyfparser.filepath import FilePath
from dyfparser.utils.read import read_conllu
import os
from datetime import datetime
from dyfparser.train.train import train
import torch

if __name__ == '__main__':
    # m_parser : <class 'utils.parser_utils.Parser'>
    # embeddings : <class 'numpy.ndarray'>
    # embeddings.shape : (m_parser.n_tokens,50)
    # 某些单词的词向量(源自en-cw.txt)  或  满足高斯分布的随机数向量
    parser = Parser(FilePath.train_file, FilePath.embedding_file, 1000)
    dev_set = read_conllu(FilePath.dev_file, True, 500)
    test_set = read_conllu(FilePath.test_file, True, 500)
    output_dir = os.path.join(FilePath.results_dir,
                              "{:%Y%m%d_%H%M%S}/".format(datetime.now()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir + "model.weights"
    # lr = learning rate
    # 训练结束后将model weights存入目标文件
    train(parser, parser.get_trainset(), dev_set, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    parser.model.load_state_dict(torch.load(output_path))
    print("Final evaluation on test set")
    parser.model.eval()
    UAS, dependencies = parser.parse(test_set)
    parser.save(FilePath.default_parser_file, FilePath.default_weight_file)
