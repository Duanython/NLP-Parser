from dyfparser.utils.toolkit import isbool_or_exception


class ParserConfig(object):

    @classmethod
    def defaultConfig(cls):
        return cls()

    def __init__(self, language='english',
                 with_punct=True,
                 unlabeled=True,
                 lowercase=False,
                 use_pos=True):
        self.__use_dep = True
        self.language = language
        self.with_punct = with_punct
        self.unlabeled = unlabeled
        self.lowercase = lowercase
        self.use_pos = use_pos

    def tolist(self):
        return [self.language, self.with_punct,
                self.unlabeled, self.lowercase, self.use_pos]

    @property
    def with_punct(self):
        return self.__with_punct

    @with_punct.setter
    def with_punct(self, value):
        isbool_or_exception(value)
        self.__with_punct = value

    @property
    def unlabeled(self):
        return self.__unlabeled

    @unlabeled.setter
    def unlabeled(self, value):
        isbool_or_exception(value)
        self.__unlabeled = value
        self.__use_dep = self.__use_dep and (not self.__unlabeled)

    @property
    def lowercase(self):
        return self.__lowercase

    @lowercase.setter
    def lowercase(self, value):
        isbool_or_exception(value)
        self.__lowercase = value

    @property
    def use_pos(self):
        return self.__use_pos

    @use_pos.setter
    def use_pos(self, value):
        isbool_or_exception(value)
        self.__use_pos = value

    @property
    def use_dep(self):
        return self.__use_dep
