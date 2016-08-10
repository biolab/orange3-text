===========
POS Tagging
===========

.. figure:: icons/postagging.png

Marks documents with part-of-speech tags.

Signals
-------

**Inputs**:

-  **Corpus**

Corpus instance.

**Outputs**:

-  **Corpus**

Marked corpus.

Description
-----------

This widget adds POS tags to each token in documents. These tags could be used in BagOfWords widget to distinguish
tokens with the same spelling.

Stanford POS Tagger
^^^^^^^^^^^^^^^^^^^

This widget also allows you to load `Stanford POS Taggers <http://nlp.stanford.edu/software/tagger.shtml>`_.
Download tagger you are interested in and provide path to model along with ``stanford-postagger.jar``.

*Note:* ensure that `Java DK <http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`_
is properly installed.

Example
-------

No example yet for this widget.
