class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """

        self.sentence = sentence
        self.stack = ["ROOT"]
        self.buffer = sentence[:]
        self.dependencies = []

    def parse_step(self, transition):
        # shift
        if transition == "S":
            word = self.buffer.pop(0)
            self.stack.append(word)
        # left arc
        elif transition == "LA":
            # self.stack[-1] -> self.stack[-2]
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack.pop(-2)
        # right arc
        else:
            # self.stack[-2] -> self.stack[-1]
            # arrow 右边的单词弹出
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            self.stack.pop()

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    # 句子与PartialParse实例一一对应
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    # n句话，n个PartialParse
    unfinished_parses = partial_parses[:]
    n = len(unfinished_parses)
    while n > 0:
        l = min(n, batch_size)
        # l个预测对应l句话的下一步动作
        transitions = model.predict(unfinished_parses[:l])
        # 每句话和那句话相应的预测动作
        for parse, trans in zip(unfinished_parses[:l], transitions):
            parse.parse_step(trans)
            if len(parse.stack) == 1:
                unfinished_parses.remove(parse)
                n -= 1
    return [partial_parses.dependencies for partial_parses in partial_parses]
