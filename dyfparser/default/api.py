import spacy
from spacy import displacy

from dyfparser.parser.parser import Parser
from dyfparser.utils.read import read_doc
excludes = ['parser', 'ner', 'attribute_ruler', 'lemmatizer']

# nlp.pipe_names:['tok2vec', 'tagger', 'senter']
nlp = spacy.load('en_core_web_sm', exclude=excludes)
nlp.enable_pipe('senter')
# parser
parser = Parser.default_parser()



# visualize options
options = {
    'arrow_stroke': 1,
    'arrow_width': 7,
    'arrow_spacing': 10,
    'word_spacing': 25,
    'bg': '#f7f1da',
}


def dyfparsing(text: str, visualize=False):
    """

    :param text: 自然语言语句或自然语言文本
    :param visualize: 是否可视化
    :return: 依存关系边集合
    """
    dependencies = parser.parsing(read_doc(nlp(text)))
    if visualize:
        displacy.serve(dependencies, manual=True, host='localhost', options=options)
    return dependencies
