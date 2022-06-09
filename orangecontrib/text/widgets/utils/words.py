from typing import Iterable

from Orange.data import StringVariable, Table, Domain

WORDS_COLUMN_NAME = "Words"


def create_words_table(words: Iterable) -> Table:
    words_var = StringVariable(WORDS_COLUMN_NAME)
    words_var.attributes = {"type": "words"}
    domain = Domain([], metas=[words_var])
    data = [[w] for w in words]
    words = Table.from_list(domain, data)
    words.name = "Words"
    return words
