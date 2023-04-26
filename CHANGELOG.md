Change Log
==========

[next] - TBA
------------
* ...

[1.13.0] - 2023-04-26
--------------------
##### Enhancements
* Sentiment Analysis - Language from corpus ([#954](../../pull/954))
* NYTimes - add language to corpus ([#926](../../pull/926))
* Document embedding - Use language from the corpus ([#953](../../pull/953))
* Guardian - infer language and add to corpus ([#925](../../pull/925))
* Score Documents - Use SBERT embedding instead of FastText ([#930](../../pull/930))
* Wikipedia - add language to corpus ([#928](../../pull/928))
* Keywords - replace embedding with MBERT ([#932](../../pull/932))
* Remove elements with delete/backspace key ([#948](../../pull/948))
* Semantic Viewer - show document when no words at the input ([#933](../../pull/933))
* PubMed - add language to corpus ([#927](../../pull/927))
* Score Documents - enable matching n-grams ([#935](../../pull/935))
* Create Corpus - add language to corpus ([#924](../../pull/924))
* Twitter - add language to corpus ([#921](../../pull/921))
* Import documents - language dialog and language guessing ([#918](../../pull/918))
* Add language to corpus ([#916](../../pull/916))

##### Bugfixes
* Corpus - Fix contexts to be compatible between sessions ([#966](../../pull/966))
* Keywords: Fix selection and use idClicked instead of buttonClicked ([#965](../../pull/965))
* Ontology - Show labels instead of names for imported ontolgies ([#936](../../pull/936))
* Corpus widget - preserve corpus's preprocessing ([#950](../../pull/950))
* Pubmed - replace deprecated extend_corpus ([#949](../../pull/949))
* Corpus - preserve name in extend_attributes ([#937](../../pull/937))
* Make widgets PyQt6 compatible ([#929](../../pull/929))
* Corpus: Unpickle corpus without language ([#919](../../pull/919))
* Score Documents - cast bool scores to float ([#913](../../pull/913))
* Annotator: Invalidate clusters on setting change ([#910](../../pull/910))

[1.12.0] - 2022-10-06
--------------------
##### Bugfixes
* Ontology - remove cache and other fixes ([#896](../../pull/896))
* VectorizationComputeValue - fix unpickling old pickles ([#904](../../pull/904))
* Keywords/Score Documents - fix ctrl/cmd selection ([#902](../../pull/902))
* Word Enrichment - fix PyQt6 incompatibility and sort imports ([#901](../../pull/901))
* VectorizationComputeValue - remove wrongly set "original" variable ([#900](../../pull/900))

[1.11.0] - 2022-08-24
--------------------
##### Enhancements
* Ontology widget documentation ([#881](../../pull/881))
* Collocations widget ([#782](../../pull/782))

##### Bugfixes
* Replace exec_ with exec and fix deprecations ([#887](../../pull/887))
* Ontology - fix cannot be called from a running event loop error ([#882](../../pull/882))

[1.10.0] - 2022-07-08
--------------------
##### Enhancements
* Document Embedding - set SBERT as default ([#875](../../pull/875))
* Document Embedding: add SBERT ([#839](../../pull/839))

##### Bugfixes
* Corpus: fix ngrams_corpus ([#871](../../pull/871))

[1.9.0] - 2022-06-21
--------------------
##### Bugfixes
* Import Documents - fix metadata matching ([#870](../../pull/870))
* Word cloud - add type to the selected words output ([#868](../../pull/868))

[1.8.1] - 2022-06-20
--------------------
##### Bugfixes
* Score Documents - adapt to the latest changes in document embedding ([#866](../../pull/866))
* Temporary proxy fix

[1.8.0] - 2022-06-09
--------------------
##### Enhancements
* Create corpus - new widget ([#854](../../pull/854))
* Computation in separate thread for base vectorizer; use base vectorizer for embedding ([#852](../../pull/852))
* Mark "Words" outputs as non-dynamic ([#855](../../pull/855))
* Corpus refactoring ([#767](../../pull/767))
* Functionalities for computing concave hull around clusters of points ([#816](../../pull/816))

##### Bugfixes
* Normalize - fix unpickling for Normalizers before caching was implemented ([#838](../../pull/838))
* Credential manager dialog at import ([#857](../../pull/857))
* NLTK - use proxy when proxy adresses entered ([#853](../../pull/853))
* Semantic search: fix wrong output when response is None ([#826](../../pull/826))
* Annotate documents: update to work with the latest changes in keywords ([#845](../../pull/845))
* Update embedder callbacks ([#842](../../pull/842))
* Keywords: Fix sending data while running ([#836](../../pull/836))
* Preprocess Text: set highest absolute frequency ([#807](../../pull/807))
* Wikipedia: make widget useable again ([#825](../../pull/825))
* Concave hull: fix cases when all points inline ([#821](../../pull/821))
* Keywords: Always invoke preprocessor __call__ ([#815](../../pull/815))

[1.7.0] - 2022-04-08
--------------------
##### Enhancements
* Twitter: adapt to APIv2 ([#780](../../pull/780))

##### Bugfixes
* Semantic Viewer: Put NaNs last when sorting ([#800](../../pull/800))
* Twitter: Update for Python 3.10 ([#793](../../pull/793))

[1.6.2] - 2022-01-10
--------------------
##### Bugfixes
* Solved numpy 1.22 incompatibility

[1.6.1] - 2021-11-28
--------------------
##### Enhancements
* Semantic Viewer: New widget ([#748](../../pull/748))

##### Bugfixes
* score-documents: handle document titles with newlines ([#754](../../pull/754))

[1.6.0] - 2021-11-23
--------------------
##### Enhancements
* OWLDAvis ([#731](../../pull/731))
* Corpus Viewer: Run search when finished typing the query ([#740](../../pull/740))

##### Bugfixes
* Semantic Search: increase maximal allowed chunk size, fix error when timeout ([#752](../../pull/752))
* Topic Modeling: Base selection style on itemdelegates ([#751](../../pull/751))
* Topic - remove topic computations in chunk which results in poor topics ([#750](../../pull/750))
* Semantic Search - fix callback to return proportions ([#747](../../pull/747))
* BoW: use training weights on test data ([#745](../../pull/745))
* OWScoreDocuments: Ensure unique names on output ([#744](../../pull/744))
* Sparse2CorpusSliceable: add support for np.ndarray as key ([#737](../../pull/737))
* Store ngrams_corpus correctly ([#729](../../pull/729))

[1.5.3] - 2021-10-08
--------------------
##### Bugfixes
*  UDPipe Lemmatizer: remove self.model from pickle ([#722](../../pull/722))
* Fixes for Tweepy 4.0.0 ([#725](../../pull/725))


[1.5.1] - 2021-09-16
--------------------
##### Bugfixes
* Fixing bug with non-working Gensim library


[1.5.0] - 2021-09-13
--------------------

##### Enhancements
* Keywords: Add 'Embedding' scoring method ([#666](../../pull/666))
* OW Corpus Viewer: Add annotated corpus output ([#672](../../pull/672))
* Concordance - search in the separate thread ([#668](../../pull/668))
* OWPreprocess Text: add option to filter on POS tags ([#679](../../pull/679))
* Topic Modeling: Add topic evaluation scores ([#687](../../pull/687))
* OWStatistics: Add new statistics method ([#676](../../pull/676))
* Import Documents: Add conllu reader ([#675](../../pull/675))
* OWPreprocess Text: Add Lemmagen normalizer ([#708](../../pull/708))
* normalize: speedup preprocessing with caching ([#709](../../pull/709))
* Score documents: Document selection and selection output ([#710](../../pull/710))

##### Bugfixes
* Import Documents: Read metas as the right type ([#677](../../pull/677))
* LDA: Add random state ([#688](../../pull/688))
* Preprocess: Filter POS tags along with tokens ([#692](../../pull/692))
* Topic Modeling: Remove tags from display of topics ([#693](../../pull/693))
* Score documents: fix word preprocessing ([#707](../../pull/707))
* Make Lemmagen lemmatizer picklable ([#713](../../pull/713))
* Document Embedder: fix default language setting ([#716](../../pull/716))

[1.4.0] - 2021-05-27
--------------------

##### Enhancements
* Score documents widget ([#632](../../pull/632))
* Import documents: Import from URL ([#637](../../pull/637))
* Extract Keywords: New widget ([#644](../../pull/644))
* Word List: New widget ([#634](../../pull/634))

##### Bugfixes
* Fix infer text features ([#645](../../pull/645))
* Corpus fix from_numpy and from_list; modify widget to work with corpuses without text_features ([#627](../../pull/627))
* Corpus Viewer: Handle empty corpus ([#628](../../pull/628))
* Import Documents: Handle loading folder with no readable files ([#626](../../pull/626))

[1.3.1] - 2021-01-22
--------------------
##### Bugfixes
* Revert #592 since fastText cannot be installed on Windows ([#618](../../pull/618))

[1.3.0] - 2021-01-21
--------------------
##### Enhancements
* Sentiment Analysis: Add SentiArt method ([#605](../../pull/605))
* Preprocess Text: update documentation ([#604](../../pull/604))
* [DOC] Sentiment Analysis: custom files and documentation ([#557](../../pull/557))

##### Bugfixes
* Preprocess: Filter by absolute frequency ([#601](../../pull/601))
* Corpus: extend_attributes retain preprocessing ([#599](../../pull/599))
* owwordcloud: Work with empty token list ([#588](../../pull/588))
* Simhash: Fix error when hash function is None ([#589](../../pull/589))

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


[next]: https://github.com/biolab/orange3-text/compare/1.13.0...HEAD
[1.13.0]: https://github.com/biolab/orange3-text/compare/1.12.0...1.13.0
[1.12.0]: https://github.com/biolab/orange3-text/compare/1.11.0...1.12.0
[1.11.0]: https://github.com/biolab/orange3-text/compare/1.10.0...1.11.0
[1.10.0]: https://github.com/biolab/orange3-text/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/biolab/orange3-text/compare/1.8.1...1.9.0
[1.8.1]: https://github.com/biolab/orange3-text/compare/1.8.0...1.8.1
[1.8.0]: https://github.com/biolab/orange3-text/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/biolab/orange3-text/compare/1.6.2...1.7.0
[1.6.2]: https://github.com/biolab/orange3-text/compare/1.6.1...1.6.2
[1.6.1]: https://github.com/biolab/orange3-text/compare/1.6.0...1.6.1
[1.6.0]: https://github.com/biolab/orange3-text/compare/1.5.3...1.6.0
[1.5.3]: https://github.com/biolab/orange3-text/compare/1.5.1...1.5.3
[1.5.1]: https://github.com/biolab/orange3-text/compare/1.5.0...1.5.1
[1.5.0]: https://github.com/biolab/orange3-text/compare/1.4.0...1.5.0
[1.4.0]: https://github.com/biolab/orange3-text/compare/1.3.1...1.4.0
[1.3.1]: https://github.com/biolab/orange3-text/compare/1.3.0...1.3.1
[1.3.0]: https://github.com/biolab/orange3-text/compare/1.2.0...1.3.0
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
