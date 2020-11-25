import re
from copy import copy
from string import punctuation
from typing import Callable, List, Optional, Tuple

import numpy as np
from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget
from orangewidget.widget import Msg

from orangecontrib.text import Corpus

# those functions are implemented here since they are used in more statistics
from orangecontrib.text.preprocess import (
    LowercaseTransformer,
    Preprocessor,
    RegexpTokenizer,
    PreprocessorList)
from orangecontrib.text.widgets.utils.context import (
    AlmostPerfectContextHandler,
)


def num_words(document: str, callback: Callable) -> int:
    """
    Return number of words in document-string. Word is every entity divided by
    space, tab, newline.
    """
    callback()
    return len(document.split())


def char_count(document: str, callback: Callable) -> int:
    """
    Count number of alpha-numerical in document/string.
    """
    callback()
    return sum(c.isalnum() for c in document)


def digit_count(document: str, callback: Callable) -> int:
    """
    Count number of digits in document/string.
    """
    callback()
    return sum(c.isdigit() for c in document)


def count_appearances(
    document: str, characters: List[str], callback: Callable
) -> int:
    """
    Count number of appearances of chars from `characters` list.
    """
    callback()
    # I think it supports the majority of main languages
    # Y can be vowel too sometimes - it is not possible to distinguish
    return sum(document.lower().count(c) for c in characters)


def preprocess_only_words(corpus: Corpus) -> Corpus:
    """
    Apply the preprocessor that splits words, transforms them to lower case
    (and removes punctuations).

    Parameters
    ----------
    corpus
        Corpus on which the preprocessor will be applied.

    Returns
    -------
    Preprocessed corpus. Result of pre-processing is saved in tokens/ngrams.
    """
    p = PreprocessorList(
        [LowercaseTransformer(),
         # by default regexp keeps only words (no punctuations, no spaces)
         RegexpTokenizer()]
    )
    return p(corpus)


# every statistic returns a np.ndarray with statistics
# and list with variables names - it must be implemented here since some
# statistics in the future will have more variables


def words_count(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of words in each document.
    """
    corpus = preprocess_only_words(corpus)
    # np.c_ makes column vector (ndarray) out of the list
    # [1, 2, 3] -> [[1], [2], [3]]
    return (
        np.c_[[num_words(d, callback) for d in corpus.documents]],
        ["Word count"],
    )


def characters_count(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of characters without spaces, newlines, tabs, ...
    """
    return (
        np.c_[[char_count(d, callback) for d in corpus.documents]],
        ["Character count"],
    )


def n_gram_count(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of n-grams in every document
    """

    def ng_count(n_gram: List[str]):
        callback()
        return len(n_gram)

    return np.c_[list(map(ng_count, corpus.ngrams))], ["N-gram count"]


def average_word_len(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes word density as: word count / character count + 1
    """
    return (
        np.c_[
            [
                char_count(d, lambda: True) / num_words(d, callback)
                for d in corpus.documents
            ]
        ],
        ["Average word length"],
    )


def punctuation_count(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of punctuation signs
    """

    def num_punctuation(document: str):
        callback()
        return sum(document.count(c) for c in punctuation)

    return (
        np.c_[list(map(num_punctuation, corpus.documents))],
        ["Punctuation count"],
    )


def capital_count(
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of capital letters in documents
    """

    def num_capitals(document: str):
        callback()
        return sum(1 for c in document if c.isupper())

    return (
        np.c_[list(map(num_capitals, corpus.documents))],
        ["Capital letter count"],
    )


def vowel_count(
    corpus: Corpus, vowels: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of vowels in documents
    """
    # comma separated string of vowels to list
    vowels = [v.strip() for v in vowels.split(",")]
    return (
        np.c_[
            [count_appearances(d, vowels, callback) for d in corpus.documents]
        ],
        ["Vowel count"],
    )


def consonant_count(
    corpus: Corpus, consonants: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count number of consonants in documents. Consonants are all alnum
    characters except vowels and numbers
    """
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
    corpus: Corpus, _: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Ratio between unique words count and all words count
    """
    corpus = preprocess_only_words(corpus)

    def perc_unique(tokens: str):
        callback()
        if not tokens:
            return np.nan
        return len(set(tokens)) / len(tokens)

    return np.c_[list(map(perc_unique, corpus.tokens))], ["% unique words"]


def starts_with(
    corpus: Corpus, prefix: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that starts with the string in `prefix`.
    """
    corpus = preprocess_only_words(corpus)

    def number_starts_with(tokens: List[str]):
        callback()
        return sum(t.startswith(prefix) for t in tokens)

    return (
        np.c_[list(map(number_starts_with, corpus.tokens))],
        [f"Starts with {prefix}"],
    )


def ends_with(
    corpus: Corpus, postfix: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that ends with the string in `postfix`.
    """
    corpus = preprocess_only_words(corpus)

    def number_ends_with(tokens: List[str]):
        callback()
        return sum(t.endswith(postfix) for t in tokens)

    return (
        np.c_[list(map(number_ends_with, corpus.tokens))],
        [f"Ends with {postfix}"],
    )


def contains(
    corpus: Corpus, text: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Number of words that contains string in `text`.
    """
    return (
        np.c_[
            [count_appearances(d, [text], callback) for d in corpus.documents]
        ],
        [f"Contains {text}"],
    )


def regex(
    corpus: Corpus, expression: str, callback: Callable
) -> Tuple[np.ndarray, List[str]]:
    """
    Count occurrences of pattern in `expression`.
    """
    pattern = re.compile(expression)

    def number_regex(tokens: List[str]):
        callback()
        return sum(bool(pattern.match(t)) for t in tokens)

    return (
        np.c_[list(map(number_regex, corpus.tokens))],
        [f"Regex {expression}"],
    )


def pos_tags(
    corpus: Corpus, pos_tags: str, callback: Callable
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Count number of specified pos tags in corpus
    """
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
    """

    def __init__(self, function: Callable, pattern: str) -> None:
        self.function = function
        self.pattern = pattern

    def __call__(self, data: Corpus) -> np.ndarray:
        """
        This function compute values on new table.
        """
        # lambda is added as a placeholder for a callback.
        return self.function(data, self.pattern, lambda: True)[0]


# the definition of all statistics used in this widget, if new statistic
# is required ad it to this list

STATISTICS = [
    # (name of the statistics, function to compute, default value)
    # if default value is None - text box is not required
    ("Word count", words_count, None),
    ("Character count", characters_count, None),
    ("N-gram count", n_gram_count, None),
    ("Average word length", average_word_len, None),
    ("Punctuation count", punctuation_count, None),
    ("Capital letter count", capital_count, None),
    ("Vowel count", vowel_count, "a,e,i,o,u"),
    (
        "Consonant count",
        consonant_count,
        "b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z",
    ),
    ("Per cent unique words", per_cent_unique_words, None),
    ("Starts with", starts_with, ""),
    ("Ends with", ends_with, ""),
    ("Contains", contains, ""),
    ("Regex", regex, ""),
    ("POS tag", pos_tags, "NN,VV,JJ"),
]
STATISTICS_NAMES = list(list(zip(*STATISTICS))[0])
STATISTICS_FUNCTIONS = list(list(zip(*STATISTICS))[1])
STATISTICS_DEFAULT_VALUE = list(list(zip(*STATISTICS))[2])


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

    for s, patern in statistics:
        fun = STATISTICS_FUNCTIONS[s]
        result = fun(corpus, patern, advance)
        if result is not None:
            result = result + (ComputeValue(fun, patern),)
        state.set_partial_result((s, patern, result))


class OWStatistics(OWWidget, ConcurrentWidgetMixin):
    name = "Statistics"
    description = "Create new statistic variables for documents."
    keywords = []
    icon = "icons/Statistics.svg"

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    class Warning(OWWidget.Warning):
        not_computed = Msg(
            "{} statistics cannot be computed and is omitted from results."
        )

    want_main_area = False
    settingsHandler = AlmostPerfectContextHandler(0.9)

    # settings
    default_rules = [(0, ""), (1, "")]  # rules used to reset the active rules
    active_rules: List[Tuple[int, str]] = ContextSetting(default_rules[:])
    # rules active at time of apply clicked
    applied_rules: Optional[List[Tuple[int, str]]] = None

    result_dict = {}

    def __init__(self) -> None:
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None

        # the list with combos from the widget
        self.combos = []
        # the list with line edits from the widget
        self.line_edits = []
        # the list of buttons in front of controls that removes them
        self.remove_buttons = []

        self._init_controls()

    def _init_controls(self) -> None:
        """ Init all controls of the widget """
        self._init_statistics_box()
        box = gui.hBox(self.controlArea)
        gui.rubber(box)
        gui.button(
            box,
            self,
            "Apply",
            autoDefault=False,
            width=180,
            callback=self.apply,
        )

    def _init_statistics_box(self) -> None:
        """
        Init the statistics box in control area - place where used statistics
        are listed, remove, and added.
        """
        patternbox = gui.vBox(self.controlArea, box=True)
        self.rules_box = rules_box = QGridLayout()
        patternbox.layout().addLayout(self.rules_box)
        box = gui.hBox(patternbox)
        gui.button(
            box,
            self,
            "+",
            callback=self._add_row,
            autoDefault=False,
            flat=True,
            minimumSize=(QSize(20, 20)),
        )
        gui.rubber(box)
        self.rules_box.setColumnMinimumWidth(1, 70)
        self.rules_box.setColumnMinimumWidth(0, 10)
        self.rules_box.setColumnStretch(0, 1)
        self.rules_box.setColumnStretch(1, 1)
        self.rules_box.setColumnStretch(2, 100)
        rules_box.addWidget(QLabel("Feature"), 0, 1)
        rules_box.addWidget(QLabel("Pattern"), 0, 2)
        self.adjust_n_rule_rows()

    def adjust_n_rule_rows(self) -> None:
        """
        Add or remove lines in statistics box if needed and fix the tab order.
        """

        def _add_line():
            n_lines = len(self.combos) + 1

            # add delete symbol
            button = gui.button(
                None,
                self,
                label="×",
                flat=True,
                height=20,
                styleSheet="* {font-size: 16pt; color: silver}"
                "*:hover {color: black}",
                autoDefault=False,
                callback=self._remove_row,
            )
            button.setMinimumSize(QSize(12, 20))
            self.rules_box.addWidget(button, n_lines, 0)
            self.remove_buttons.append(button)

            # add statistics type dropdown
            combo = QComboBox()
            combo.addItems(STATISTICS_NAMES)
            combo.currentIndexChanged.connect(self._sync_edit_combo)
            self.rules_box.addWidget(combo, n_lines, 1)
            self.combos.append(combo)

            # add line edit for patern
            line_edit = QLineEdit()
            self.rules_box.addWidget(line_edit, n_lines, 2)
            line_edit.textChanged.connect(self._sync_edit_line)
            self.line_edits.append(line_edit)

        def _remove_line():
            self.combos.pop().deleteLater()
            self.line_edits.pop().deleteLater()
            self.remove_buttons.pop().deleteLater()

        def _fix_tab_order():
            # TODO: write it differently - check create class
            for i, (r, c, l) in enumerate(
                zip(self.active_rules, self.combos, self.line_edits)
            ):
                c.setCurrentIndex(r[0])  # update combo
                l.setText(r[1])  # update line edit
                if STATISTICS_DEFAULT_VALUE[r[0]] is not None:
                    l.setVisible(True)
                else:
                    l.setVisible(False)

        n = len(self.active_rules)
        while n > len(self.combos):
            _add_line()
        while len(self.combos) > n:
            _remove_line()
        _fix_tab_order()

    def _add_row(self) -> None:
        """ Add a new row to the statistic box """
        self.active_rules.append((0, ""))
        self.adjust_n_rule_rows()

    def _remove_row(self) -> None:
        """ Removes the clicked row in the statistic box """
        remove_idx = self.remove_buttons.index(self.sender())
        del self.active_rules[remove_idx]
        self.adjust_n_rule_rows()

    def _sync_edit_combo(self) -> None:
        """ Update rules when combo value changed """
        combo = self.sender()
        edit_index = self.combos.index(combo)
        selected_i = combo.currentIndex()
        default_value = STATISTICS_DEFAULT_VALUE[selected_i]
        self.active_rules[edit_index] = (
            selected_i,
            default_value or self.active_rules[edit_index][1],
        )
        self.adjust_n_rule_rows()

    def _sync_edit_line(self) -> None:
        """ Update rules when line edit value changed """
        line_edit = self.sender()
        edit_index = self.line_edits.index(line_edit)
        self.active_rules[edit_index] = (
            self.active_rules[edit_index][0],
            line_edit.text(),
        )

    @Inputs.corpus
    def set_data(self, corpus) -> None:
        self.closeContext()
        self.corpus = corpus
        self.active_rules = self.default_rules[:]
        self.openContext(corpus)
        self.adjust_n_rule_rows()
        self.result_dict = {}  # empty computational results when new data
        # reset old output - it also handle case with corpus == None
        self.Outputs.corpus.send(None)

        # summary
        if corpus:
            self.info.set_input_summary(
                len(corpus), format_summary_details(corpus)
            )
            self.apply()
        else:
            self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

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
        self, result: Tuple[int, str, Tuple[np.ndarray, List[str], Callable]]
    ) -> None:
        statistic, patern, result = result
        self.result_dict[(statistic, patern)] = result

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

        # summary
        self.info.set_output_summary(
            len(new_corpus), format_summary_details(new_corpus)
        )


if __name__ == "__main__":
    WidgetPreview(OWStatistics).run(Corpus.from_file("book-excerpts"))
