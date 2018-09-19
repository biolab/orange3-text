The Guardian
============

Fetching data from `The Guardian Open Platform <http://open-platform.theguardian.com>`_.

Inputs
    None

Outputs
    Corpus
        A collection of documents from the Guardian newspaper.


**Guardian** retrieves articles from the Guardian newspaper via their API. For the widget to work, you need to provide the API key, which you can get at `their access platform <https://open-platform.theguardian.com/access/>`_.

.. figure:: images/Guardian-stamped.png
   :scale: 50%

1. Insert the API key for the widget to work.

.. figure:: images/Guardian-API.png
   :scale: 50%

2. Provide the query and set the time frame from which to retrieve the articles.
3. Define which features to retrieve from the Guardian platform.
4. Information on the output.
5. Press *Search* to start retrieving the articles or *Stop* to stop the retrieval.

Example
-------

**Guardian** can be used just like any other data retrieval widget in Orange, namely :doc:`NY Times <nytimes>`, :doc:`Wikipedia <wikipedia>`, :doc:`Twitter <twitter>` or :doc:`PubMed <pubmed>`.

We will retrieve 240 articles mentioning *slovenia* between september 2017 and september 2018. The text will include article headline and content. Upon pressing *Search*, the articles will be retrieved.

We can observe the results in the :doc:`Corpus Viewer <corpusviewer>` widget.

.. figure:: images/Guardian-Example.png
