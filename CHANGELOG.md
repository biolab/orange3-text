Change Log
==========

[next] - TBA
------------
* ...

[1.2.0] - 2020-10-12
--------------------
##### Enhancements
* Document embedders: additional languages ([#565](../../pull/565))
* Corpus Viewer: Output selected data and memorize selection ([#562](../../pull/562))
* List files that are not loaded ([#560](../../pull/560))

##### Bugfixes
* Corpus - from_table: keep text feature when renamed ([#585](../../pull/585))
* Corpus - extend attributes: Fix error with renamed text feature ([#574](../../pull/574))
* Import documents: normalize imported text and file names ([#568](../../pull/568))
* Corpus to network ([#559](../../pull/559))
* Corpus: fix deprecated use of array ([#563](../../pull/563))
* Preprocess: Retain corpus ids ([#553](../../pull/553))

[1.1.0] - 2020-08-07
--------------------
##### Bugfixes
* Preprocess: Retain corpus ids ([#553](../../pull/553))

[1.0.0] - 2020-06-12
--------------------
##### Enhancements
* Refactor preprocessors ([#506](../../pull/506))

##### Bugfixes
* Twitter: Fix errors ([#538](../../pull/538))
* Update ulr to use https for udpipe models ([#524](../../pull/524))

[0.9.1] - 2020-05-05
--------------------

[0.9.0] - 2020-04-29
--------------------
##### Enhancements
* Word Enrichment: compute in separate thread ([#492](../../pull/492))
* Bag of Words: option to show bow features ([#499](../../pull/499))
* Word Cloud: threaded ([#502](../../pull/502))
* Corpus: remove unnecessary empty values ([#505](../../pull/505))
* Statistics widget: new widget for feature construction ([#503](../../pull/503))
* Document Embedding widget: word embedding for documents ([#504](../../pull/504))
* Corpus to Network widget: widget for computing networks from documents ([#509](../../pull/509))

##### Bugfixes
* Word Cloud: don't show zero weights ([#501](../../pull/501))

[0.8.0] - 2020-02-01
--------------------
##### Enhancements
* Add Concordance output as text feature. ([#476](../../pull/476))
* Corpus: add Title dropdown. ([#481](../../pull/481))
* Word Cloud: show bow weights ([#486](../../pull/486))
* Topic Modeling: change output to emulate PyLDAvis ([#483](../../pull/483))
* Corpus: make unique titles ([#490](../../pull/490))
* Word Cloud: major rewrite (spacing, tests, bugfixes) ([#493](../../pull/493))

##### Bugfixes
* Remove typing as dependency ([#475](../../pull/475))
* Topic Modeling: select topic is schema-only setting ([#478](../../pull/478))
* Include widgets in coverage ([#487](../../pull/487))
* Corpus: use DomainContextHandler ([#491](../../pull/491))

[0.7.4] - 2019-11-11
------------------
##### Bugfixes
* Remove ufal.udpipe-temp dependency ([#467](../../pull/467))
* Remove redundant webview in Word Cloud ([#464](../../pull/464))
* Rename GeoMap to Document Map ([#458](../../pull/458))
* Fix stopword filtering ([#456](../../pull/456))
* Pickling UDPipe models ([#447](../../pull/447))
* Remove validate_email dependency ([#442](../../pull/442))

[0.7.3] - 2019-07-09
------------------

* Fix broken import ([#448](../../pull/448))

[0.7.2] - 2019-07-02
------------------

* Compatibility with split Orange Canvas ([#445](../../pull/445))

[0.7.1] - 2019-06-18
------------------

* Remove google-compute-engine dependency ([#439](../../pull/439))

[0.7.0] - 2019-06-14
------------------
##### Enhancements
* Word Enrichment: FDR imported from Orange ([#416](../../pull/416))

##### Bugfixes
* OWPreprocess: autocommit when changing n-grams ([#436](../../pull/436))
* Import Documents: sanitize pdfs ([#435](../../pull/435))
* OWPreprocess: fix Stanford model loading ([#419](../../pull/419))
* Corpus Viewer: mark filtered text with Python ([#408](../../pull/408))
* Topic Modelling: HDP shows no topics ([#413](../../pull/413))
* Fix PyQt to 5.11 ([#410](../../pull/410))
* Guardian: properly display Unicode characters ([#406](../../pull/406))

[0.6.0] - 2018-12-06
------------------
##### Enhancements
* UDPipe models work offline ([#394](../../pull/394))
* Word Cloud: scale visualization ([#381](../../pull/381))

##### Bugfixes
* Duplicate Detection: output corresponding duplicate cluster ([#379](../../pull/379))
* Update Twitter API calls to retrieve extended tweets ([#383](../../pull/383))
* Twitter: fix retrieving author timeline ([#389](../../pull/389))


[0.5.1] - 2018-09-27
------------------
##### Bugfixes
* Make ufal.udpipe work on all platforms ([#384](../../pull/384))
* Documentation can appear in canvas ([#376](../../pull/376))


[0.5.0] - 2018-09-14
--------------------
##### Enhancements
* Concordance: output concordances ([#371](../../pull/371))
* UDPipe Lemmatizer ([#367](../../pull/367))
* OWWordCloud: Add the word cloud image to the report ([#364](../../pull/364))
* Sentiment Analysis: add Slovenian language ([#366](../../pull/366))

##### Bugfixes
* Guardian: Handle API limit exception ([#365](../../pull/365))
* Corpus Viewer: fix highlighting ([#375](../../pull/375))


[0.4.0] - 2018-07-23
--------------------
##### Enhancements
* Improved report functionality ([#357](../../pull/357))
* OWPreprocess: Add choice in FilteringModule for All Files (*) ([#334](../../pull/334))

##### Bugfixes
* Concordance: clear selection on changing word ([#353](../../pull/353))
* CorpusViewer: Break long urls ([#310](../../pull/310))
* OWPreprocess: no stopword files on Windows ([#355](../../pull/355))
* Bag of words: Use vectorized 'BINARY' local weighting ([#342](../../pull/342))
* Corpus: X and Y cast as float ([#330](../../pull/330))
* Bag of words: work on document with no tokens ([#356](../../pull/356))


[0.3.0] - 2017-12-05
--------------------
##### Enhancements
* Corpus & Bow: Improve sparsity handling according to Orange>=3.8.0 ([#281](../../pull/281))
* Download NLTK data asynchronously ([#304](../../pull/304))
* Add Table Input to Corpus ([#308](../../pull/308))

##### Bugfixes
* Corpus: Remove text features which not in metas ([#325](../../pull/325))
* Topic Modelling: Do not call get all topics table when no corpus ([#322](../../pull/322))
* Concordance: Selection settings ([#249](../../pull/249))
* Preprocess: Use default tokenizer when None ([#294](../../pull/294))


[0.2.5] - 2017-07-27
--------------------


[0.2.4] - 2017-06-04
--------------------


[0.2.3] - 2017-02-08
--------------------


[0.2.2] - 2016-12-06
--------------------


[0.2.1] - 2016-10-12
--------------------


[0.2.0] - 2016-09-15
--------------------


[0.1.11] - 2016-06-26
--------------------


[0.1.10] - 2016-04-01
--------------------


[0.1.9] - 2015-12-14
--------------------


[0.1.8] - 2015-12-11
--------------------


[0.1.7] - 2015-12-02
--------------------


[0.1.6] - 2015-12-01
--------------------


[0.1.5] - 2015-10-26
--------------------


[0.1.4] - 2015-09-10
--------------------


[0.1.3] - 2015-09-09
--------------------


[next]: https://github.com/biolab/orange3-text/compare/1.2.0...HEAD
[1.2.0]: https://github.com/biolab/orange3-text/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/biolab/orange3-text/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/biolab/orange3-text/compare/0.9.1...1.0.0
[0.9.1]: https://github.com/biolab/orange3-text/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/biolab/orange3-text/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/biolab/orange3-text/compare/0.7.4...0.8.0
[0.7.4]: https://github.com/biolab/orange3-text/compare/0.7.3...0.7.4
[0.7.3]: https://github.com/biolab/orange3-text/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/biolab/orange3-text/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/biolab/orange3-text/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/biolab/orange3-text/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/biolab/orange3-text/compare/0.5.1...0.6.0
[0.5.1]: https://github.com/biolab/orange3-text/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/biolab/orange3-text/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/biolab/orange3-text/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/biolab/orange3-text/compare/0.2.5...0.3.0
[0.2.5]: https://github.com/biolab/orange3-text/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/biolab/orange3-text/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/biolab/orange3-text/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/biolab/orange3-text/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/biolab/orange3-text/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/biolab/orange3-text/compare/0.1.11...0.2.0
[0.1.11]: https://github.com/biolab/orange3-text/compare/0.1.10...0.1.11
[0.1.10]: https://github.com/biolab/orange3-text/compare/0.1.9...0.1.10
[0.1.9]: https://github.com/biolab/orange3-text/compare/0.1.8...0.1.9
[0.1.8]: https://github.com/biolab/orange3-text/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/biolab/orange3-text/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/biolab/orange3-text/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/biolab/orange3-text/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/biolab/orange3-text/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/biolab/orange3-text/commits/0.1.3
