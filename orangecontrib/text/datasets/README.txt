The following files originate from the Ana Cardoso Cachopo's Homepage:
[http://ana.cachopo.org/datasets-for-single-label-text-categorization]
and primarily focus on single-topic text categorization.
    * 20ng-test-all-terms.txt   [20Newsgroups dataset]
    * 20ng-train-all-terms.txt  [20Newsgroups dataset]
    * r8-test-all-terms.txt     [Reuters-21578 dataset]
    * r8-train-all-terms.txt    [Reuters-21578 dataset]
    * r52-test-all-terms.txt    [Reuters-21578 dataset]
    * r52-train-all-terms.txt   [Reuters-21578 dataset]

About the data sets:
    * 20Newsgroups dataset: This dataset is a collection of approximately 20,000 newsgroup documents, partitioned
                            (nearly) evenly across 20 different newsgroups.
    * Reuters-21578 dataset: These documents appeared on the Reuters newswire in 1987 and were manually classified
                             by personnel from Reuters Ltd.
    * r8: only 8 most frequent topics
    * r52: all 52 topics found in documents with only one label

About the preprocessing:
    * all-terms: Obtained from the original datasets by applying the following transformations:
                    * Substitute TAB, NEWLINE and RETURN characters by SPACE.
                    * Keep only letters (that is, turn punctuation, numbers, etc. into SPACES).
                    * Turn all letters to lowercase.
                    * Substitute multiple SPACES by a single SPACE.
                    * The title/subject of each document is simply added in the beginning of the document's text.
-----------------------------------------------------------------------------------------------------------------------
Dataset: Book excerpts

In this example we are trying to separate text written for children from that written
for adults using multidimensional scaling. There are 140 texts divided into two categories:
children and adult. The texts are all in English.
-----------------------------------------------------------------------------------------------------------------------
Dataset: Deerwester

Small data set of 9 documents, 5 about human-computer interaction and 4 about graphs.
This data set originates from the paper about Latent Semantic Analysis [1].

[1] Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by Latent Semantic
    Analysis. Journal of the American Society for Information Science, 41(6), 391–407.
-----------------------------------------------------------------------------------------------------------------------
Dataset: Friends Transcripts

Transcripts from the Friends series. Originating from http://home.versatel.nl/friendspic0102/
