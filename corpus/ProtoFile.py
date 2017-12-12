import os
import pickle
from collections import namedtuple, Counter

import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize.moses import MosesTokenizer
from tqdm import tqdm

import config as cfg
import features_config as feat_cfg

import io
import logging
from builtins import any as b_any

from preprocessing.feature_engineering.GeniaTagger import GeniaTagger
from preprocessing.feature_engineering.pos import PosTagger
import html

logger = logging.getLogger(__name__)

Tag = namedtuple("Tag", "tag_id, tag_name, start, end, words")
Link = namedtuple("Link", "l_id, l_name, arg1, arg2")


# its a good idea to keep a datastructure like
# list of sentences, where each sentence is a list of words : [[word1, word2, word3,...], [word1, word2...]]

class ProtoFile:
    def __init__(self, filename, genia=None, gen_features=False, to_filter=False):
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.protocol_name = self.basename
        self.text_file = self.filename + '.txt'
        self.ann_file = self.filename + '.ann'

        with io.open(self.text_file, 'r', encoding='utf-8', newline='') as t_f, io.open(self.ann_file, 'r',
                                                                                        encoding='utf-8',
                                                                                        newline='') as a_f:
            self.tokenizer = MosesTokenizer()
            self.lines = t_f.readlines()  # list of strings, each string is a sentence
            self.text = "".join(self.lines)  # full text
            self.ann = a_f.readlines()
            self.status = self.__pretest()
            self.links = []

        if self.status:
            sents = [self.tokenizer.tokenize(line) for line in self.lines]  # generate list of list of words
            self.heading = sents[0]
            self.sents = sents[1:]
            self.tags = self.__parse_tags()
            self.unique_tags = set([tag.tag_name for tag in self.tags])
            self.__std_index()
            self.__parse_links()
            self.tag_0_id = 'T0'
            self.tag_0_name = 'O'
            self.tokens2d = self.gen_tokens(labels_allowed=cfg.LABELS)
            self.tokens2d = [[self.clean_html_tag(token) for token in token1d] for token1d in self.tokens2d]
            self.word_cnt = sum(len(tokens1d) for tokens1d in self.tokens2d)
            self.f_df = None
            if gen_features:
                if genia:
                    self.pos_tags = self.__gen_pos_genia(genia)
                else:
                    self.pos_tags = self.__gen_pos_stanford()
                self.conll_deps = self.__gen_dep()

            if to_filter:
                self.filter()

    @staticmethod
    def clean_html_tag(token):
        token.word = html.unescape(token.word)
        return token

    def filter(self):
        # tokens2d, pos_tags, and conll_deps are filtered if a sentence was not tagged
        new_tokens2d = []
        new_pos_tags = []
        new_conll_deps = []
        for tokens1d, pos_tag1d, deps1d in zip(self.tokens2d, self.pos_tags, self.conll_deps):
            # here tokens1d is a sentence of word sequences, and label is a sequence of labels for a sentence.

            # check if any of the labels in this sentence have POSITIVE_LABEL in them, if they do, then consider that
            # sentence, else discard that sentence.
            if b_any(cfg.POSITIVE_LABEL in token.label for token in tokens1d):
                new_tokens2d.append(tokens1d)
                new_pos_tags.append(pos_tag1d)
                new_conll_deps.append(deps1d)

        self.tokens2d = new_tokens2d
        self.pos_tags = new_pos_tags
        self.conll_deps = new_conll_deps

    def get_deps(self):
        return [nltk.DependencyGraph(conll_dep, top_relation_label='root') for conll_dep in self.conll_deps]

    def __gen_dep(self):
        d_cache = os.path.join(cfg.DEP_PICKLE_DIR, self.protocol_name + '.p')
        try:
            # loading saved dep parsers
            conll_deps = pickle.load(open(d_cache, 'rb'))

        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            dep = StanfordDependencyParser(path_to_jar=feat_cfg.STANFORD_PARSER_JAR,
                                           path_to_models_jar=feat_cfg.STANFORD_PARSER_MODEL_JAR,
                                           java_options="-mx3000m")

            # adding pos data to dep parser speeds up dep generation even further
            dep_graphs = [sent_dep for sent_dep in dep.tagged_parse_sents(self.pos_tags)]

            # save dependency graph in conll format
            conll_deps = [next(deps).to_conll(10) for deps in dep_graphs]

            pickle.dump(conll_deps, open(d_cache, 'wb'))

        return conll_deps

    def __gen_pos_genia(self, pos_tagger):
        p_cache = os.path.join(cfg.POS_GENIA_DIR, self.protocol_name + '.p')
        try:
            pos_tags = pickle.load(open(p_cache, 'rb'))
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pos_tags = pos_tagger.parse_through_file([" ".join(sent) for sent in self.sents])

            pickle.dump(pos_tags, open(p_cache, 'wb'))
        return pos_tags

    def __gen_pos_stanford(self):
        pos = PosTagger(feat_cfg.STANFORD_POS_JAR_FILEPATH, feat_cfg.STANFORD_MODEL_FILEPATH,
                        cache_filepath=None)
        p_cache = os.path.join(cfg.POS_PICKLE_DIR, self.protocol_name + '.p')
        try:
            pos_tags = pickle.load(open(p_cache, 'rb'))
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pos_tags = pos.tag_sents(self.sents)
            # for some reason stanford parser deletes words that are just underscores, and
            # dependency parser cannot deal with an empty text in pos tagger, so the below hack.
            pos_tags = [[pos_tag if pos_tag[0] else ('_', pos_tag[1]) for pos_tag in p1d] for p1d in pos_tags]
            pickle.dump(pos_tags, open(p_cache, 'wb'))

        return pos_tags

    def cnt_words(self):
        if self.status:
            w = sum([len(sent) for sent in self.sents[1:]])
            return w

            # generic counter of entities, supported by a function callback fn that depends on tag's properties

    # ent_counter returns a dict = {'ENTITY1' : summation for all tags of 'ENTITY1'(fn(tag))}
    def __ent_counter(self, ent_types, fn):
        tag_cnts = dict()
        if self.status:
            for tag in self.tags:
                if tag.tag_name in ent_types:
                    if tag.tag_name in tag_cnts:
                        tag_cnts[tag.tag_name] += fn(tag)
                    else:
                        tag_cnts[tag.tag_name] = fn(tag)

        return tag_cnts

    def ent_cnt(self, ent_types):
        tag_cnts = self.__ent_counter(ent_types, lambda x: 1)

        return tag_cnts

    def ent_w_cnt(self, ent_types):
        def cnt_words(tag):
            string = tag.word
            words = nltk.word_tokenize(string)
            return len(words)

        tag_cnts = self.__ent_counter(ent_types, cnt_words)

        return tag_cnts

    # calculates the total no of chars (including spaces) for each entity in a protocol file
    def ent_span_len(self, ent_types):
        tag_cnts = self.__ent_counter(ent_types, lambda x: len(x.word))

        return tag_cnts

    def __std_index(self):
        # modifies the ann text such that all
        # Exx Action:Txx Using:Exx convert to
        # Exx Action:Txx Using:Tyy
        # so that they are easier to resolve later
        # given that Txx can be independently resolved,
        # whereas Exx sometimes have forward and backward dependencies
        def search_tag(e_id):
            if e_id[0] == 'E':
                for _line in self.ann:
                    if _line.find(e_id) == 0:
                        logging.info(_line.rstrip())
                        spl = _line.split()
                        return spl[1].split(':')[1]
            else:
                return e_id

        def replace_Es(string):
            if string[0] == 'E':
                sp_res = string.split()
                front_half = sp_res[0]
                args = [tuple(sp.split(':')) for sp in sp_res[1:]]
            elif string[0] == 'R':
                sp_res = string.split()
                r_id = sp_res[0]
                r_name = sp_res[1]
                front_half = " ".join([r_id, r_name])
                args = [tuple(sp.split(':')) for sp in sp_res[2:]]
            else:
                # nothing to replace
                return string

            # args = [(Action, Txx), (Using, Exx)]
            replaced_args = [(rel_name, search_tag(tid)) for rel_name, tid in args]
            # replaced_args = [(Action, Txx), (Using, Txx)]
            args_str = " ".join([":".join(item) for item in replaced_args])
            # args_str = "Action:Txx Using:Txx"

            string = " ".join([front_half, args_str])
            # string = "Exx Action:Txx Using:Txx"

            return string

        for i, line in enumerate(self.ann):
            self.ann[i] = replace_Es(line)

    def __pretest(self):
        """
        Returns false if annotation file or text file is empty
        :return:
        """
        if len(self.lines) < 2:
            logger.debug(self.sents)
            return False
        if len(self.ann) < 1:
            logger.debug(self.ann)
            return False
        return True

    def __parse_links(self):
        if self.links:
            logger.error("Already parsed, I am not parsing again")
            return
        for line in [t for t in self.ann if (t[0] == 'E' or t[0] == 'R')]:
            if line[0] == 'E':
                e = self.__parse_e(line)
                self.links.extend(e)
            elif line[0] == 'R':
                r = self.__parse_r(line)
                self.links.append(r)

    def get_tag_by_id(self, tid):
        if tid[0] == 'T':
            ret = [tag for tag in self.tags if tag.tag_id == tid]
        else:
            ret = [link.arg1 for link in self.links if link.l_id == tid]
        return ret[0]

    def __parse_e(self, e):
        links = []
        temp = e.rstrip()
        temp = temp.split()
        e_id = temp[0]
        arg1_id = temp[1].split(':')[1]

        arg1_tag = self.get_tag_by_id(arg1_id)
        if temp[2:]:
            for rel in temp[2:]:
                r_name, arg2_id = rel.split(':')
                arg2_tag = self.get_tag_by_id(arg2_id)
                links.append(Link(e_id, r_name, arg1_tag, arg2_tag))

        return links

    def __parse_r(self, r):
        r_id, r_name, arg1, arg2 = r.rstrip().split()
        arg1_id = arg1.split(':')[1]
        arg2_id = arg2.split(':')[1]
        link = Link(r_id, r_name, self.get_tag_by_id(arg1_id), self.get_tag_by_id(arg2_id))
        return link

    def __parse_tags(self):
        tags = []
        only_tags = [t for t in self.ann if t[0] == 'T']
        for tag in only_tags:
            tag = tag.rstrip()
            temp = tag.split('\t')

            if len(temp[1].split()) == 3:
                tag_name, start, end = temp[1].split()
            elif len(temp[1].split()) == 4:
                tag_name, start, _, end = temp[1].split()
            else:
                tag_name, start, _, _, end = temp[1].split()

            t = Tag(tag_id=temp[0],
                    tag_name=tag_name,
                    start=int(start),
                    end=int(end),
                    words=self.tokenizer.tokenize(temp[2]))

            tags.append(t)
        return tags

    @staticmethod
    def __contain(s1, e1, s2, e2):
        if s2 <= s1 and e1 <= e2:
            return True
        elif not (s2 >= s1 and e2 >= e1 or s2 <= s1 and e2 <= e1):
            logger.debug("partial overlap: {0} {1} {2} {3}".format(s1, e1, s2, e2))
            return False
        return False

    @staticmethod
    def make_bio(tag):
        # returns [(word, label), (word, label)]
        # where the label is encoded with B, I, or O based on its position in the tag
        # tag = Tag(tag_id, tag_name, start, end, words)

        labels = ['B-' + tag.tag_name]
        labels += ['I-' + tag.tag_name for _ in tag.words[1:]]
        return list(zip(tag.words, labels))

    def get_tag_by_start(self, start):
        for tag in self.tags:
            if tag.start == start:
                return tag

        logging.debug("Protocol={0}: No tag found with start == {1}".format(self.protocol_name, start))
        return None

    def gen_tokens(self, labels_allowed=None):
        # for a list of list of words returns a list of list of tokens
        # [[Token(word, label), Token(word, label)], [Token(word, label), Token(word, label)]]
        # BIO encoding

        start = len(self.sents[0])
        ret = []
        for sent in self.sents:
            word_label_pairs = []
            wi = 0
            # for every word in the sentence
            while wi < len(sent):
                word = sent[wi]
                start = self.text.find(word, start)

                tag = self.get_tag_by_start(start)
                # we dont want tags that are not allowed
                if labels_allowed:
                    if tag and tag.tag_name not in labels_allowed:
                        tag = None

                if tag:
                    # tag was found
                    # make bio returns a list of (word, label) pairs which we extend to the word_label_pairs list.
                    logging.debug("Protocol={0}: Tag was found = {1}".format(self.protocol_name, tag))
                    word_label_pairs.extend(self.make_bio(tag))
                    start = tag.end
                    wi += len(tag.words)

                if not tag:
                    # its likely that there is no tag for this word
                    logging.debug("Protocol={0}: Tag was not found at word = {1}".format(self.protocol_name, word))
                    word_label_pairs.append((word, 'O'))
                    start += len(word)
                    wi += 1

            tokens = [Token(word, label) for word, label in word_label_pairs]

            ret.append(tokens)

        return ret

    # ###################interface from Article#############################################
    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.

        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear. (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        if add_no_ne_label:
            counts = Counter([token.label for token in self.tokens2d])
        else:
            counts = Counter([token.label for token in self.tokens2d \
                              if token.label != cfg.NO_NE_LABEL])
        return counts.most_common()

    def count_labels(self, add_no_ne_label=False):
        """Returns how many named entity tokens appear in the article/document.

        Args:
            add_no_ne_label: Whether to also count unlabeled words. (Default is False.)
        Returns:
            Count of all named entity tokens (integer).
        """
        return sum([count[1] for count in self.get_label_counts(add_no_ne_label=add_no_ne_label)])


class Token(object):
    """Encapsulates a token/word.
    Members:
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See Window.apply_features().)
    """

    def __init__(self, word, label=cfg.NO_NE_LABEL):
        """Initialize a new Token object.
        Args:
            original: The original word as found in the text document, including the label,
                e.g. "foo", "John/PER".
        """
        # self.original = original

        self.word = word
        self.label = label
        # self._word_ascii = None
        self.feature_values = None


if __name__ == '__main__':
    pro = ProtoFile("./simple_input/protocol_235")
    [print([(token.word, token.label) for token in tokens1d]) for tokens1d in pro.tokens2d]