from os.path import join as j


class FilePath(object):
    root_dir = r'D:\Projects\dyfparser'
    data_dir = j(root_dir, 'data')
    doc_dir = j(root_dir, 'doc')
    results_dir = j(root_dir, 'results')
    train_file = j(data_dir, 'conllu', 'train.conllu')
    dev_file = j(data_dir, 'conllu', 'dev.conllu')
    test_file = j(data_dir, 'conllu', 'test.conllu')
    embedding_file = j(data_dir, 'wordvec', 'en-cw')
    default_weight_file = j(data_dir, 'weights', 'default.weights')
    default_parser_file = j(data_dir, 'parser', 'default.parser')
