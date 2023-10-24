import re
from collections import Counter
from copy import copy
from itertools import groupby
from string import punctuation
from typing import Callable, List, Optional, Tuple, Union, Iterator, Dict

import numpy as np
from AnyQt.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit, QSizePolicy

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget
from nltk import tokenize
from orangecanvas.gui.utils import disconnected
from orangewidget.widget import Msg

from orangecontrib.text import Corpus


class Sources:
    DOCUMENTS = "Documents"
    TOKENS = "Tokens"  # tokens or ngrams - depending on statistic


def num_words(document: Union[str, List], callback: Callable) -> int:
    """
    Return number of words in document-string. Word is every entity divided by
    space, tab, newline.
    """
    callback()
    if isinstance(document, str):
        document = document.split()
    return len(document)


def char_count(document: Union[str, List], callback: Callable) -> int:
    """
    Count number of alpha-numerical in document/string.
    """
    callback()
    if isinstance(document, List):
        document = "".join(document)
    return sum(c.isalnum() for c in document)


def digit_count(document: str, callback: Callable) -> int:
    """
    Count number of digits in document/string.
    """
    callback()
    return sum(c.isdigit() for c in document)


def count_appearances(
    document: Union[str, List], characters: List[str], callback: Callable
) -> int:
    """
    Count number of appearances of chars from `characters` list.
    """
    callback()
    # I think it supports the majority of main languages
    # Y can be vo wel too sometimes - it is not possible to distinguish
    if isinstance(document, str):
        return sum(document.lower().count(c) for c in characters)
    else:
        return sum(d.lower().count(c) for c in characters for d in document)


def get_source(corpus: Corpus, source: str) -> Union[List[str], Iterator[List[str]]]:
    """
    Extract source from corpus according to source variable:
    - if source == Sources.DOCUMENTS return documents
    - if source == Sources.TOKENS return ngrams
    """
    if source == Sources.DOCUMENTS:
        return corpus.documents
    elif source == Sources.TOKENS:
        return corpus.ngrams
    else:
        raise ValueError(f"Wrong source {source}")


# every statistic returns a np.ndarray with statistics
# and list with variables names - it must be implemented here since some
# statistics in the future will have more variables


def words_count(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of words in each document.
    """
    assert source == Sources.DOCUMENTS
    # np.c_ makes column vector (ndarray) out of the list
    # [1, 2, 3] -> [[1], [2], [3]]
    return np.c_[[num_words(d, callback) for d in corpus.documents]], ["Word count"]


def characters_count(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of characters without spaces, newlines, tabs, ...
    """
    source = get_source(corpus, source)
    return np.c_[[char_count(d, callback) for d in source]], ["Character count"]


def n_gram_count(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of n-grams in every document
    """
    assert source == Sources.TOKENS

    def ng_count(n_gram: List[str]):
        callback()
        return len(n_gram)

    return np.c_[list(map(ng_count, corpus.ngrams))], ["N-gram count"]


def average_word_len(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes word density as: word count / character count + 1
    """
    source = get_source(corpus, source)
    return (
        np.c_[[char_count(d, lambda: True) / num_words(d, callback) for d in source]],
        ["Average word length"],
    )


def punctuation_count(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of punctuation signs
    """
    assert source == Sources.DOCUMENTS

    def num_punctuation(document: str):
        callback()
        return sum(document.count(c) for c in punctuation)

    return (
        np.c_[list(map(num_punctuation, corpus.documents))],
        ["Punctuation count"],
    )


def capital_count(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of capital letters in documents
    """
    assert source == Sources.DOCUMENTS

    def num_capitals(document: str):
        callback()
        return sum(1 for c in document if c.isupper())

    return (
        np.c_[list(map(num_capitals, corpus.documents))],
        ["Capital letter count"],
    )


def vowel_count(
    corpus: Corpus, vowels: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of vowels in documents
    """
    assert source == Sources.DOCUMENTS

    # comma separated string of vowels to list
    vowels = [v.strip() for v in vowels.split(",")]
    return (
        np.c_[
            [count_appearances(d, vowels, callback) for d in corpus.documents]
        ],
        ["Vowel count"],
    )


def consonant_count(
    corpus: Corpus, consonants: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of consonants in documents. Consonants are all alnum
    characters except vowels and numbers
    """
    assert source == Sources.DOCUMENTS

    # comma separated string of consonants to list
    consonants = [v.strip() for v in consonants.split(",")]
    return (
        np.c_[
            [
                count_appearances(d, consonants, callback)
                for d in corpus.documents
            ]
        ],
        ["Consonant count"],
    )


def per_cent_unique_words(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Ratio between unique words count and all words count
    """
    assert source == Sources.TOKENS

    def perc_unique(tokens: str):
        callback()
        if not tokens:
            return np.nan
        return len(set(tokens)) / len(tokens)

    return np.c_[list(map(perc_unique, corpus.ngrams))], ["% unique words"]


def starts_with(
    corpus: Corpus, prefix: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that starts with the string in `prefix`.
    """
    assert source == Sources.TOKENS

    def number_starts_with(tokens: List[str]):
        callback()
        return sum(t.startswith(prefix) for t in tokens)

    return (
        np.c_[list(map(number_starts_with, corpus.ngrams))],
        [f"Starts with {prefix}"],
    )


def ends_with(
    corpus: Corpus, postfix: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that ends with the string in `postfix`.
    """
    assert source == Sources.TOKENS

    def number_ends_with(tokens: List[str]):
        callback()
        return sum(t.endswith(postfix) for t in tokens)

    return (
        np.c_[list(map(number_ends_with, corpus.ngrams))],
        [f"Ends with {postfix}"],
    )


def contains(
    corpus: Corpus, text: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that contains string in `text`.
    """
    source = get_source(corpus, source)
    return (
        np.c_[[count_appearances(d, [text], callback) for d in source]],
        [f"Contains {text}"],
    )


def regex(
    corpus: Corpus, expression: str, source: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count occurrences of pattern in `expression`.
    """
    pattern = re.compile(expression)

    def regex_matches(text: Union[str, List]):
        callback()
        if isinstance(text, str):
            return len(re.findall(pattern, text))
        else:
            return sum(len(re.findall(pattern, ngram)) for ngram in text)

    source = get_source(corpus, source)
    return np.c_[list(map(regex_matches, source))], [f"Regex {expression}"]


def pos_tags(
    corpus: Corpus, pos_tags: str, source: str, callback: Callable
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Count number of specified pos tags in corpus
    """
    assert source == Sources.TOKENS
    p_tags = [v.strip().lower() for v in pos_tags.split(",")]

    def cust_count(tags):
        callback()
        tags = [t.lower() for t in tags]
        return sum(tags.count(t) for t in p_tags)

    if corpus.pos_tags is None:
        return None
    return (
        np.c_[[cust_count(p) for p in corpus.pos_tags]],
        [f"POS tags {pos_tags}"],
    )


def yule(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Yule's I measure: higher number is higher diversity - richer vocabulary
    PSP volume 42 issue 2 Cover and Back matter. (1946).
    Mathematical Proceedings of the Cambridge Philosophical Society, 42(2), B1-B2.
    doi:10.1017/S0305004100022799
    """
    assert source == Sources.TOKENS
    if corpus.pos_tags is None:
        return None

    def yules_i(tags):
        callback()
        d = Counter(tags)
        m1 = float(len(d))
        m2 = sum([len(list(g)) * (freq ** 2) for freq, g in
                  groupby(sorted(d.values()))])
        try:
            return (m1 * m1) / (m2 - m1)
        except ZeroDivisionError:
            return 0

    return (
        np.c_[[yules_i(p) for p in corpus.pos_tags]],
        [f"Yule's I"],
    )


def lix(
    corpus: Corpus, _: str, source: str, callback: Callable
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Readability index LIX
    https://en.wikipedia.org/wiki/Lix_(readability_test)
    """
    assert source == Sources.TOKENS
    tokenizer = tokenize.PunktSentenceTokenizer()

    def lix_index(document, tokens):
        callback()
        # if the text is a single sentence, scores will be high
        sentences = len(tokenizer.tokenize(document))
        words = len(tokens)
        long_words = len([token for token in tokens if len(token) > 6])
        try:
            return words/sentences + (long_words*100/words)
        except ZeroDivisionError:
            return 0

    return (
        np.c_[[lix_index(d, tokens) for d, tokens in zip(corpus.documents,
                                                         corpus.tokens)]],
        ["LIX index"],
    )


class ComputeValue:
    """
    Class which provides compute value functionality. It stores the function
    that is used to compute values on new data table using this domain.

    Attributes
    ----------
    function
        Function that computes new values
    pattern
        Some statistics need additional parameter with the pattern
        (e.g. starts with), for others it is set to empty string.
    source
        Part of the corpus used for computation: either tokens/ngrams or whole documents
    """

    def __init__(self, function: Callable, pattern: str, source: str) -> None:
        self.function = function
        self.pattern = pattern
        self.source = source

    def __call__(self, data: Corpus) -> np.ndarray:
        """
        This function compute values on new table.
        """
        # lambda is added as a placeholder for a callback.
        return self.function(data, self.pattern, self.source, lambda: True)[0]

    def __eq__(self, other):
        return self.function == other.function and self.pattern == other.pattern

    def __hash__(self):
        return hash((self.function, self.pattern))


# the definition of all statistics used in this widget, if new statistic
# is required ad it to this list

STATISTICS = [
    # (name of the statistics, function to compute, default value)
    # if default value is None - text box is not required
    ("Word count", words_count, None, (Sources.DOCUMENTS,)),
    ("Character count", characters_count, None, (Sources.DOCUMENTS, Sources.TOKENS)),
    ("N-gram count", n_gram_count, None, (Sources.TOKENS,)),
    ("Average term length", average_word_len, None, (Sources.DOCUMENTS, Sources.TOKENS)),
    ("Punctuation count", punctuation_count, None, (Sources.DOCUMENTS,)),
    ("Capital letter count", capital_count, None, (Sources.DOCUMENTS,)),
    ("Vowel count", vowel_count, "a,e,i,o,u", (Sources.DOCUMENTS,)),
    (
        "Consonant count",
        consonant_count,
        "b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z",
        (Sources.DOCUMENTS,),
    ),
    ("Per cent unique terms", per_cent_unique_words, None, (Sources.TOKENS,)),
    ("Starts with", starts_with, "", (Sources.TOKENS,)),
    ("Ends with", ends_with, "", (Sources.TOKENS,)),
    ("Contains", contains, "", (Sources.DOCUMENTS, Sources.TOKENS)),
    ("Regex", regex, "", (Sources.DOCUMENTS, Sources.TOKENS)),
    ("POS tag", pos_tags, "NN,VV,JJ", (Sources.TOKENS,)),
    ("Yule's I", yule, None, (Sources.TOKENS,)),
    ("LIX index", lix, None, (Sources.TOKENS,)),
]
STATISTICS_NAMES = list(list(zip(*STATISTICS))[0])
STATISTICS_FUNCTIONS = list(list(zip(*STATISTICS))[1])
STATISTICS_DEFAULT_VALUE = list(list(zip(*STATISTICS))[2])
STATISTICS_DEFAULT_SOURCES = list(list(zip(*STATISTICS))[3])


def run(corpus: Corpus, statistics: Tuple[int, str], state: TaskState) -> None:
    """
    This function runs the computation for new features.
    All results will be reported as a partial results.

    Parameters
    ----------
    corpus
        The corpus on which the computation is held.
    statistics
        Tuple of statistic pairs to be computed:
        (statistics id, string pattern)
    state
        State used to report progress and partial results.
    """
    # callback is called for each corpus element statistics time
    tick_values = iter(np.linspace(0, 100, len(corpus) * len(statistics)))

    def advance():
        state.set_progress_value(next(tick_values))

    for s, patern, source in statistics:
        fun = STATISTICS_FUNCTIONS[s]
        result = fun(corpus, patern, source, advance)
        if result is not None:
            result = result + (ComputeValue(fun, patern, source),)
        state.set_partial_result((s, patern, source, result))


class OWStatistics(OWWidget, ConcurrentWidgetMixin):
    name = "Statistics"
    description = "Create new statistic variables for documents."
    keywords = "statistics"
    icon = "icons/Statistics.svg"

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    class Warning(OWWidget.Warning):
        not_computed = Msg(
            "{} statistics cannot be computed and is omitted from results."
        )

    # todo: update settings version and migration
    want_main_area = False
    mainArea_width_height_ratio = None

    settings_version = 2
    # rules used to reset the active rules
    default_rules = [(0, "", STATISTICS[0][-1][0]), (1, "", STATISTICS[0][-1][0])]
    active_rules: List[Tuple[int, str, str]] = Setting(default_rules[:])
    # rules active at time of apply clicked
    applied_rules: Optional[List[Tuple[int, str]]] = None

    result_dict = {}

    def __init__(self) -> None:
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None

        # the list with combos for selecting statistics from the widget
        self.statistics_combos = []
        # the list with line edits from the widget
        self.line_edits = []
        # the list of buttons in front of controls that removes them
        self.remove_buttons = []
        # the list with combos for selecting on what statistics computes
        self.source_combos = []

        self._init_controls()

    def _init_controls(self) -> None:
        """ Init all controls of the widget """
        self._init_statistics_box()

        gui.button(self.buttonsArea, self, "Apply", callback=self.apply)

    def get_button(self, label, callback):
        return

    def _init_statistics_box(self) -> None:
        """
        Init the statistics box in control area - place where used statistics
        are listed, remove, and added.
        """
        box = gui.vBox(self.controlArea, box=True)

        rules_box = gui.vBox(box)
        self.rules_grid = grid = QGridLayout()
        rules_box.layout().addLayout(self.rules_grid)
        grid.setColumnMinimumWidth(1, 100)
        grid.setColumnMinimumWidth(0, 25)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 100)
        grid.addWidget(QLabel("Feature"), 0, 1)
        grid.addWidget(QLabel("Pattern"), 0, 2)
        grid.addWidget(QLabel("Compute for"), 0, 3)

        gui.button(
            box,
            self,
            "+",
            callback=self._add_row,
            autoDefault=False,
            width=34,
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum),
        )
        gui.rubber(box)

        self.adjust_n_rule_rows()

    def adjust_n_rule_rows(self) -> None:
        """
        Add or remove lines in statistics box if needed and fix the tab order.
        """

        def _add_line():
            n_lines = len(self.statistics_combos) + 1

            # add delete symbol
            button = gui.button(
                None, self, "×", callback=self._remove_row,
                addToLayout=False, autoDefault=False, width=34,
                sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum))
            self.rules_grid.addWidget(button, n_lines, 0)
            self.remove_buttons.append(button)

            # add statistics type dropdown
            combo = QComboBox()
            combo.addItems(STATISTICS_NAMES)
            combo.currentIndexChanged.connect(self._sync_edit_combo)
            self.rules_grid.addWidget(combo, n_lines, 1)
            self.statistics_combos.append(combo)

            # add line edit for pattern
            line_edit = QLineEdit()
            self.rules_grid.addWidget(line_edit, n_lines, 2)
            line_edit.textChanged.connect(self._sync_edit_line)
            self.line_edits.append(line_edit)

            # add statistics type dropdown
            combo = QComboBox()
            combo.currentIndexChanged.connect(self._sync_edit_source_combo)
            self.rules_grid.addWidget(combo, n_lines, 3)
            self.source_combos.append(combo)

        def _remove_line():
            self.statistics_combos.pop().deleteLater()
            self.line_edits.pop().deleteLater()
            self.source_combos.pop().deleteLater()
            self.remove_buttons.pop().deleteLater()

        def _fix_tab_order():
            for i, (r, c, l, s) in enumerate(
                zip(self.active_rules, self.statistics_combos, self.line_edits, self.source_combos)
            ):
                c.setCurrentIndex(r[0])  # update combo
                l.setText(r[1])  # update line edit
                if STATISTICS_DEFAULT_VALUE[r[0]] is not None:
                    l.setVisible(True)
                else:
                    l.setVisible(False)
                with disconnected(s.currentIndexChanged, self._sync_edit_source_combo):
                    s.clear()
                    items = STATISTICS_DEFAULT_SOURCES[r[0]]
                    s.addItems(items)
                    s.setCurrentText(r[2])
                    s.setDisabled(len(items) == 1)

        n = len(self.active_rules)
        while n > len(self.statistics_combos):
            _add_line()
        while len(self.statistics_combos) > n:
            _remove_line()
        _fix_tab_order()

    def _add_row(self) -> None:
        """ Add a new row to the statistic box """
        self.active_rules.append((0, "", STATISTICS_DEFAULT_SOURCES[0][0]))
        self.adjust_n_rule_rows()

    def _remove_row(self) -> None:
        """ Removes the clicked row in the statistic box """
        remove_idx = self.remove_buttons.index(self.sender())
        del self.active_rules[remove_idx]
        self.adjust_n_rule_rows()

    def _sync_edit_combo(self) -> None:
        """ Update rules when combo value changed """
        combo = self.sender()
        edit_index = self.statistics_combos.index(combo)
        selected_i = combo.currentIndex()
        default_value = STATISTICS_DEFAULT_VALUE[selected_i] or ""
        default_source = STATISTICS_DEFAULT_SOURCES[selected_i][0]
        self.active_rules[edit_index] = (selected_i, default_value, default_source)
        self.adjust_n_rule_rows()

    def _sync_edit_line(self) -> None:
        """ Update rules when line edit value changed """
        line_edit = self.sender()
        edit_index = self.line_edits.index(line_edit)
        arules = self.active_rules[edit_index]
        self.active_rules[edit_index] = (arules[0], line_edit.text(), arules[2])

    def _sync_edit_source_combo(self) -> None:
        """ Update rules when source value change """
        combo = self.sender()
        edit_index = self.source_combos.index(combo)
        value = combo.currentText()
        arules = self.active_rules[edit_index]
        self.active_rules[edit_index] = (arules[0], arules[1], value)

    @Inputs.corpus
    def set_data(self, corpus) -> None:
        self.corpus = corpus
        self.adjust_n_rule_rows()
        self.result_dict = {}  # empty computational results when new data
        # reset old output - it also handle case with corpus == None
        self.Outputs.corpus.send(None)
        self.apply()

    def apply(self) -> None:
        """
        This function is called when user click apply button. It starts
        the computation. When computation is finished results are shown
        on the output - on_done.
        """
        if self.corpus is None:
            return
        self.applied_rules = copy(self.active_rules)
        self.cancel()  # cancel task since user clicked apply again
        rules_to_compute = [
            r for r in self.active_rules if r not in self.result_dict
        ]
        self.start(run, self.corpus, rules_to_compute)

    def on_exception(self, exception: Exception) -> None:
        raise exception

    def on_partial_result(
        self, result: Tuple[int, str, str, Tuple[np.ndarray, List[str], Callable]]
    ) -> None:
        statistic, patern, source, result = result
        self.result_dict[(statistic, patern, source)] = result

    def on_done(self, result: None) -> None:
        # join results
        if self.corpus:
            self.output_results()

        # remove unnecessary results from dict - it can happen that user
        # already removes the statistic from gui but it is still computed
        for k in list(self.result_dict.keys()):
            if k not in self.active_rules:
                del self.result_dict[k]

    def output_results(self) -> None:
        self.Warning.not_computed.clear()
        to_stack = []
        attributes = []
        comput_values = []
        not_computed = []
        for rule in self.applied_rules:
            # check for safety reasons - in practice should not happen
            if rule in self.result_dict:
                res = self.result_dict[rule]
                if res is None:
                    not_computed.append(STATISTICS_NAMES[rule[0]])
                else:
                    data, variables, comp_value = res
                    to_stack.append(data)
                    attributes += variables
                    comput_values.append(comp_value)
        if not_computed:
            self.Warning.not_computed(", ".join(not_computed))
        new_corpus = self.corpus.extend_attributes(
            np.hstack(to_stack) if to_stack else np.empty((len(self.corpus), 0)),
            attributes, compute_values=comput_values
        )
        self.Outputs.corpus.send(new_corpus)

    @classmethod
    def migrate_settings(cls, settings: Dict, version: int):
        def def_source(idx):
            """Return source that behaviour is the most similar to previous version"""
            if STATISTICS_NAMES[idx] == "Regex":
                # regex was working on tokens in the previous version
                return Sources.TOKENS
            # others that allow both sources were working on documents
            return STATISTICS_DEFAULT_SOURCES[idx][0]

        if version < 2:
            if "active_rules" in settings:
                new_rules = [(r, v, def_source(r)) for r, v in settings["active_rules"]]
                settings["active_rules"] = new_rules


if __name__ == "__main__":
    WidgetPreview(OWStatistics).run(Corpus.from_file("book-excerpts"))
