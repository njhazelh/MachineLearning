from enum import Enum, auto

def read_sentences(filename):
    sentences = []
    with open(filename, 'r') as f:
        words = []
        for line in f:
            line = line.strip()
            if len(line) != 0:
                words.append(line)
            else:
                sentences.append(words)
                words = []
    return sentences

def read_test_soln(filename):
    sentences = []
    parts = []
    with open(filename, 'r') as f:
        words = []
        word_parts = []
        for line in f:
            line = line.strip()
            if len(line) != 0:
                word, tag = line.split()
                words.append(word)
                word_parts.append(WordTag.resolve(tag))
            else:
                sentences.append(words)
                parts.append(word_parts)
                words = []
                word_parts = []
    return sentences, parts

class WordTag(Enum):
    ADJ = auto()
    ADV = auto()
    INTJ = auto()
    NOUN = auto()
    PROPN = auto()
    VERB = auto()
    ADP = auto()
    AUX = auto()
    CONJ = auto()
    DET = auto()
    NUM = auto()
    PART = auto()
    PRON = auto()
    SCONJ = auto()
    PUNCT = auto()
    SYM = auto()
    X = auto()
    STOP = auto()
    START = auto()

    @classmethod
    def resolve(cls, name):
        if name == "*":
            return cls.START
        else:
            return cls[name]

    def __repr__(self):
        return self.name

class NGram(Enum):
    ONE = 1
    TWO = 2
    THREE = 3

    @classmethod
    def resolve(cls, string):
        if string == "1-GRAM":
            return cls.ONE
        elif string == "2-GRAM":
            return cls.TWO
        elif string == "3-GRAM":
            return cls.THREE
        else:
            raise ValueError(string)

def add_word(word_tags, count, rest):
    tag, word = rest.split(" ")
    tag = WordTag[tag]
    if word in word_tags:
        if tag in word_tags[word]:
            raise RuntimeError("%s already exists for %s" % (tag, word))
        word_tags[word][tag] = int(count)
    else:
        word_tags[word] = {tag: int(count)}

def add_gram(grams, gram_size, count, rest):
    gram_size = NGram.resolve(gram_size).value
    gram = grams[gram_size]
    parts = rest.split(" ")
    for part in parts[:len(parts)-1]:
        part = WordTag.resolve(part)
        if part not in gram:
            gram[part] = {}
        gram = gram[part]
    gram[WordTag.resolve(parts[-1])] = int(count)

def read_training_data(filename):
    word_tags = {}
    grams = {1: {}, 2: {}, 3: {}}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            count, aspect, rest = line.split(" ", 2)
            if aspect == "WORDTAG":
                add_word(word_tags, count, rest)
            elif aspect in {"1-GRAM", "2-GRAM", "3-GRAM"}:
                add_gram(grams, aspect, count, rest)
            else:
                raise RuntimeError("Line Aspect not recognized %s" % aspect)

    return word_tags, grams

def write_output(sentences, tags, filename):
    if len(sentences) != len(tags):
        print("len(sentences) = %d != len(tags) = %d" % (len(sentences), len(tags)))
        exit()
    with open(filename, 'w') as f:
        for i in range(len(sentences)):
            if len(sentences[i]) != len(tags[i]):
                print("len(%s) != len(%s)" % (sentences[i], tags[i]))
                continue
            for j in range(len(sentences[i])):
                f.write("%s %s\n" % (sentences[i][j], tags[i][j].name))
            f.write("\n")
