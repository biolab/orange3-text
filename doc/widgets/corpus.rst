======
Corpus
======

.. figure:: icons/corpus.png

Load a corpus of text documents, (optionally) tagged with categories.

Signals
-------

**Inputs**:

-  (None)

**Outputs**:

-  **Corpus**

A :ref:`Corpus` instance.

Description
-----------

**Corpus** widget reads text corpora from files and sends a corpus instance to its output channel.
History of the most recently opened files is maintained in the widget.
The widget also includes a directory with sample corpora that come
pre-installed with the add-on.

The widget reads data from Excel (**.xlsx**), comma-separated (**.csv**) and
native tab-delimited (**.tab**) files.

.. figure:: images/Corpus-stamped.png

1. Browse through previously opened data files, or load any of the
   sample ones.
2. Browse for a data file.
3. Reloads currently selected data file.
4. Information on the loaded data set.
5. Features that will be used in text analysis.
6. Features that won't be used in text analysis and serve as labels or class.

You can drag and drop features between the two boxes and also change the order in which they appear.

Example
-------

The first example shows a very simple use of **Corpus** widget. Place **Corpus** onto canvas and connect
it to :doc:`Corpus Viewer <corpusviewer>`. We've used *booxexcerpts.tab* data set, which comes with the
add-on, and inspected it in **Corpus Viewer**.

.. figure:: images/Corpus-Example1.png

The second example demonstrates how to quickly visualize your corpus with :doc:`Word Cloud <wordcloud>`.
We could connect **Word Cloud** directly to **Corpus**, but instead we decided to apply some preprocessing
with :doc:`Preprocess Text <preprocesstext>`. We are again working with *bookexcerpts.tab*. We've
put all text to lowercase, tokenized (split) the text to words only, filtered out English stopwords and selected a 100 most frequent tokens.

.. figure:: images/Corpus-Example2.png
