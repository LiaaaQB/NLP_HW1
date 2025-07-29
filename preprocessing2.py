from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1

# TODO: add features to help distinguish NN/JJ
# index of tag/word?
# index of (word, tag) pairs?
# beginning of sentence/end of sentence


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f108", "f109", "f110", 'f111', "f112", "f113"]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """

        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # f100
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    #initialize f101 with known suffix:
                    known_suffix = [('ful', 'JJ'), ('ish', 'JJ'), ('like', 'JJ'), ('ous', 'JJ'), ('able', 'JJ'), ('ive', 'JJ'),
                                    ('ible' 'JJ'), ('er', 'JJR'), ('est', 'JJS'), ('ion', 'NN'), ('acy', 'NN'),
                                    ('ence', 'NN'), ('ance', 'NN'), ('hood', 'NN'), ('ism', 'NN'), ('ist', 'NN'),
                                    ('ment', 'NN'), ('ness', 'NN'), ('ity', 'NN'), ('dom', 'NN'), ('ify', 'VB'),
                                    ('ize', 'VB'), ('en', 'VB'), ('ily', 'RB')]
                    for item in known_suffix:
                        suf = item[0]
                        t = item[1]
                        self.feature_rep_dict["f101"][(suf, t)] = 5

                    # f101 + f102
                    if len(cur_word) >= 4:
                        # Suffix features in f101
                        for length in [2, 3, 4]:
                            suffix = cur_word[-length:]
                            if (suffix, cur_tag) not in self.feature_rep_dict["f101"]:
                                self.feature_rep_dict["f101"][(suffix, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f101"][(suffix, cur_tag)] += 1

                        # Prefix features in f102
                        for length in [2, 3]:
                            prefix = cur_word[:length]
                            if (prefix, cur_tag) not in self.feature_rep_dict["f102"]:
                                self.feature_rep_dict["f102"][(prefix, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f102"][(prefix, cur_tag)] += 1

                    # f108 - capital letters
                    # if cur_word[0].isupper():
                    #     if (cur_tag) not in self.feature_rep_dict["f108"]:
                    #         self.feature_rep_dict["f108"][(cur_tag)] = 1
                    #     else:
                    #         self.feature_rep_dict["f108"][(cur_tag)] += 1

                    # f109 - numbers

                    if has_numbers(cur_word):
                        if (cur_tag) not in self.feature_rep_dict["f109"]:
                            self.feature_rep_dict["f109"][(cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f109"][(cur_tag)] += 1

                    # f110 - length
                    # if (len(cur_word), cur_tag) not in self.feature_rep_dict["f110"]:
                    #     self.feature_rep_dict["f110"][(len(cur_word), cur_tag)] = 1
                    # else:
                    #     self.feature_rep_dict["f110"][(len(cur_word), cur_tag)] += 1

                    # f113 - hyphen
                    self.feature_rep_dict["f113"][('JJ')] = 10
                    if '-' in cur_word:
                        if (cur_tag) not in self.feature_rep_dict["f113"]:
                            self.feature_rep_dict["f113"][(cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f113"][(cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence)):
                    tag = sentence[i][1]
                    if not self.is_a_tag(tag):
                        continue

                    p_tag = sentence[i-1][1] if i > 3 else 'not_tag'
                    pp_tag = sentence[i-2][1] if i > 4 else 'not_tag'
                    n_word = sentence[i+1][0] if i < len(sentence) - 2 else 'not_tag'
                    p_word = sentence[i-1][0] if i > 3 else 'not_tag'

                    # f103 trigram
                    if p_tag != 'not_tag' and pp_tag != 'not_tag':
                        if (pp_tag, p_tag, tag) not in self.feature_rep_dict["f103"]:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, tag)] = 1
                        else:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, tag)] += 1

                    # f104 bigram
                    self.feature_rep_dict["f104"][('DT', 'NN')] = 10
                    if p_tag != 'not_tag':
                        if (p_tag, tag) not in self.feature_rep_dict["f104"]:
                            self.feature_rep_dict["f104"][(p_tag, tag)] = 1
                        else:
                            self.feature_rep_dict["f104"][(p_tag, tag)] += 1

                    # f105 unigram
                    if tag not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][tag] = 1
                    else:
                        self.feature_rep_dict["f105"][tag] += 1

                    # f106 (prev word, tag)
                    self.feature_rep_dict["f106"][('be', 'JJ')] = 10
                    if p_word != 'not_tag':
                        if (p_word, tag) not in self.feature_rep_dict["f106"]:
                            self.feature_rep_dict["f106"][(p_word, tag)] = 1
                        else:
                            self.feature_rep_dict["f106"][(p_word, tag)] += 1

                    # f107 (next word, tag)
                    if n_word != 'not_tag':
                        if (n_word, tag) not in self.feature_rep_dict["f107"]:
                            self.feature_rep_dict["f107"][(n_word, tag)] = 1
                        else:
                            self.feature_rep_dict["f107"][(n_word, tag)] += 1

                    # f111 beginning of sentence tag
                    # if i == 2:
                    #     if (tag) not in self.feature_rep_dict["f111"]:
                    #         self.feature_rep_dict["f111"][(tag)] = 1
                    #     else:
                    #         self.feature_rep_dict["f111"][(tag)] += 1

                    # f112 end of sentence tag
                    # if i == len(sentence) - 2:
                    #     if (tag) not in self.feature_rep_dict["f112"]:
                    #         self.feature_rep_dict["f112"][(tag)] = 1
                    #     else:
                    #         self.feature_rep_dict["f112"][(tag)] += 1

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0], i)

                    self.histories.append(history)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(self.histories)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # raise RuntimeError("oops")

    def is_a_tag(self, tag):
        """
        checks if a string is a potential Penn Treebank tag.
        :param tag: string
        :return: T/F
        """
        return all(c.isalpha() or c == '$' for c in tag)


def has_numbers(s):
    return any(char.isdigit() for char in s)





class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(),
            "f101": OrderedDict(),
            "f102": OrderedDict(),
            "f103": OrderedDict(),
            "f104": OrderedDict(),
            "f105": OrderedDict(),
            "f106": OrderedDict(),
            "f107": OrderedDict(),
            "f108": OrderedDict(),
            "f109": OrderedDict(),
            "f110": OrderedDict(),
            "f111": OrderedDict(),
            "f112": OrderedDict(),
            "f113": OrderedDict()
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix
        # print(self.small_matrix.)
        # print(self.big_matrix)
    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")
        # print(self.feature_to_idx['f100'][('The', 'DT')])
        # if ('The', 'DT') in self.feature_to_idx['f100']:
        #     print('Ani Shava')

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)





def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features (idx) that are relevant to the given history
    """
    # c_word- current word, p- previous, pp- previous previous, n- next word.
    c_word = history[0]
    c_tag = history[1]
    p_word = history[2]
    p_tag = history [3]
    pp_word = history[4]
    pp_tag = history[5]
    n_word = history[6]
    #index = history[7]
    features = []
    chosen_features = []

    # f100 (word_tag pairs)
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    #f101 (suffix)
    suffix_2, suffix_3, suffix_4 = '', '', ''
    if len(c_word) > 2: suffix_2 = c_word[-2:]
    if len(c_word) > 3: suffix_3 = c_word[-3:]
    if len(c_word) > 4: suffix_4 = c_word[-4:]
    suffixes = [suffix_2, suffix_3, suffix_4]
    for suffix in suffixes:
        if (suffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(suffix, c_tag)])


    # f102 (prefix)
    suffix_2, suffix_3, suffix_4 = '', '', ''
    if len(c_word) > 2: suffix_2 = c_word[:2]
    if len(c_word) > 3: suffix_3 = c_word[:3]
    if len(c_word) > 4: suffix_4 = c_word[:4]
    suffixes = [suffix_2, suffix_3, suffix_4]
    for suffix in suffixes:
        if (suffix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(suffix, c_tag)])

    tag = c_tag
    # f103 (trigram)
    if (pp_tag, p_tag, tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, tag)])

    # f104 (bigram)
    if (p_tag, tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, tag)])

    # f105 (unigram)
    if (tag) in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][(tag)])

    # f106 (previous word and current tag pairs)
    if (p_word, tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, tag)])

    # f107 (next word and current tag pairs)
    if (n_word, tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, tag)])

    # f108 (uppercase tags found)
    # if c_word[0].isupper():
    #     if (c_tag) in dict_of_dicts["f108"]:
    #         features.append(dict_of_dicts["f108"][(c_tag)])

    # f109 (number tags found)
    if has_numbers(c_word):
        if (c_tag) in dict_of_dicts["f109"]:
            features.append(dict_of_dicts["f109"][(c_tag)])

    # # f110 (length of word and tag pairs)
    # if (len(c_word), tag) in dict_of_dicts["f110"]:
    #     features.append(dict_of_dicts["f110"][(len(c_word), tag)])

    # f111 (beginning of sentence):
    # if p_word == '*' and pp_word == '*':
    #     if (c_tag) in dict_of_dicts["f111"]:
    #         features.append(dict_of_dicts["f111"][(c_tag)])
    #
    # # f112 (ending of sentence):
    # if n_word == '~':
    #     if (c_tag) in dict_of_dicts["f111"]:
    #         features.append(dict_of_dicts["f111"][(c_tag)])

    # f113 (hyphen)
    if '-' in c_word and tag in dict_of_dicts["f113"]:
        features.append(dict_of_dicts["f113"][(tag)])



    return features

def f101(suffix: str, c_tag: str):
    """
    returns 1 if the suffix is a suffix usually recognized with the c_tag according to the english language.
    f101: suffix of len <= 4
    :param suffix: suffix of the current word
    :param c_tag: current tag
    :return: binary 1/0
    """
    # AS SEEN IN TRAINING DATA! DON'T ACT A FOOL DOG
    # JJ (Adj.) : al, ful, ic, ish, like, ous, ate(!), able, ible
    # JJR (Adj. comperative) : er, est
    # JJS (Adj. superlative) : est
    # NN (Noun, singular) : ion, acy, age, ence, ance, hood, or, ar, ism, ist, ment, ness, ity
    # NNS (Noun, plural) : s, es
    # VB (Verb, base form) : ify, ate(!), ize, en
    # RB (Adverb) : ily
    JJ_list = ['ful', 'ish', 'like', 'ous', 'able', 'ible']
    JJR_list = ['er']
    JJS_list = ['est']
    NN_list = ['ion', 'acy', 'ence', 'ance', 'hood', 'ism', 'ist', 'ment', 'ness', 'ity', 'dom']
    VB_list = ['ify', 'ize', 'en']
    RB_list =['ily']
    if suffix in JJ_list and c_tag == 'JJ':
        return 1
    if suffix in JJR_list and c_tag == 'JJR':
        return 1
    if suffix in JJS_list and c_tag == 'JJS':
        return 1
    if suffix in NN_list and c_tag == 'NN':
        return 1
    if suffix in VB_list and c_tag == 'VB':
        return 1
    if suffix in RB_list and c_tag == 'RB':
        return 1
    return 0






def preprocess_train2(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
