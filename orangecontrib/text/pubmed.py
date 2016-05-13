import os
import re
import shelve
import warnings
import itertools
from datetime import datetime

import numpy as np
from Bio import Entrez
from Bio import Medline
from validate_email import validate_email

from Orange.canvas.utils import environ
from Orange.data import StringVariable, DiscreteVariable, Domain
from orangecontrib.text.corpus import Corpus

PUBMED_TEXT_FIELDS = [
    ('authors', 'FAU'),
    ('article_title', 'TI'),
    ('mesh_headings', 'MH'),
    ('abstract', 'AB'),
    ('pub_date', 'DP'),
]


def _join_if_list(value):
    if isinstance(value, list):
        return ' '.join(value)
    return value


def _mesh_headings_to_class(mesh_headings):
    """Extract a class from the mesh headings.

    We take the first mesh heading and extract it to use as a class.

    Args:
        mesh_headings (list): The list containing all the mesh headings.

    Returns:
        str: The class value.
    """
    # e.g. heading1/heading2,heading3/*heading4
    regex = re.compile(r'^(.\w*?)\W')
    class_mesh_groups = regex.search(mesh_headings[0])
    return class_mesh_groups.groups()[0]


def _date_to_iso(date):
    possible_date_formats = [
        '%Y %b %d',
        '%Y %b',
        '%Y',
    ]

    season_mapping = {
        'fall': 'Sep',
        'winter': 'Dec',
        'spring': 'Mar',
        'summer': 'Jun',
    }

    date = date.lower()
    # Seasons to their respective months.
    for season, month in season_mapping.items():
        date = date.replace(season, month)
    date = date.split('-')[0]  # 2015 Sep-Dec --> 2015 Sep

    for date_format in possible_date_formats:
        try:
            return datetime.strptime(date, date_format).date().isoformat()
        except ValueError:
            continue  # Try the next format.

    warnings.warn(
            'Could not parse "{}" into a date.'.format(date),
            RuntimeWarning
    )
    return None


def _records_to_corpus_entries(records, includes_metadata):
    """Create corpus entries from records.

    Args:
        records (list): A list of dictionaries that hold record data.
        includes_metadata (list): A list of tuples, where the
            elements hold the names and tags of the metadata fields that we
            wish to extract.

    Returns:
        list, list: Metadata and class values. Metadata is an array of size
            n*m, where n is the number of article records contained within
            'data' and m is the number of metadata fields we've chosen to
            extract. The variable class_values is a list, where
            the elements are class values for each article instance.
    """
    class_values = []
    metadata = np.empty((len(records), len(includes_metadata)), dtype=object)

    for row_num, record in enumerate(records):
        fields = [
                _join_if_list(record.get(field_key, ''))
                for field_name, field_key
                in includes_metadata
            ]
        fields[-1] = _date_to_iso(fields[-1])
        metadata[row_num] = np.array(fields)[None, :]

        class_values.append(_mesh_headings_to_class(record.get('MH')))
    return metadata, class_values


def _corpus_from_records(records, includes_metadata):
    """Receives PubMed records and transforms them into a corpus.

    Args:
        records (list): A list of PubMed entries.
        includes_metadata (list): A list of text fields to include.

    Returns:
        corpus: The output Corpus.
    """
    meta_values, class_values = _records_to_corpus_entries(
            records,
            includes_metadata=includes_metadata
    )
    meta_vars = [
        StringVariable.make(field_name)
        for field_name, field_key
        in includes_metadata
    ]
    class_vars = [
        DiscreteVariable(
                'section_name',
                values=list(set(class_values))
        )
    ]
    domain = Domain(
            [],
            class_vars=class_vars,
            metas=meta_vars
    )

    Y = np.array([class_vars[0].to_val(cv)
                  for cv
                  in class_values])[:, None]
    Y[np.isnan(Y)] = 0

    return Corpus(
            None,
            Y,
            meta_values,
            domain
    )


class Pubmed:

    DEFAULT_BATCH_SIZE = 1000

    def __init__(self, email, progress_callback=None, error_callback=None):
        if not validate_email(email):
            raise ValueError('{} is not a valid email address.'.format(email))

        self.email = email
        Entrez.email = email

        self.record_id_list = None  # Ids of the records available.
        self.search_record_count = 0  # Number of records.
        self.search_record_web_env = None  # BioPython history param.
        self.search_record_query_key = None  # BioPython history param.

        self.progress_callback = progress_callback
        self.error_callback = error_callback

        self.cache_path = None
        cache_folder = os.path.join(environ.buffer_dir, 'pubmedcache')
        try:
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
            self.cache_path = os.path.join(cache_folder, 'query_cache')
        except:
            warnings.warn('Could not assemble Pubmed query cache path',
                          RuntimeWarning)

    def _search_for_records(self, terms=[], authors=[],
                            pub_date_start=None, pub_date_end=None):
        """Executes a search for PubMed records.

        Results of the search, like QueryKey, WebEnv and Count are stored as
        parameters of this class's instance.

        Args:
            terms (list): Keywords to search by.
            authors (list): Authors to search by.
            pub_date_start (str): The start of the publish date chronological
                interval.
            pub_date_end (str): The end of the publish date chronological
                interval.
        """
        search_terms = []

        argument_checks = [
            {
                'argument': authors,
                'error': '{} is not a valid data type for the field "authors".'
                    .format(type(authors)),
                'conjunction': ' AND ',
                'field': 'Author',
            },
            {
                'argument': terms,
                'error': '{} is not a valid data type for the field "terms".'
                    .format(type(terms)),
                'conjunction': ' OR ',
                'field': 'All fields',
            },
        ]

        for check in argument_checks:
            if not isinstance(check.get('argument'), list):
                raise ValueError(check.get('error'))
            arguments_list = ['{}[{}]'.format(arg, check.get('field'))
                              for arg in check.get('argument')]
            arguments = check.get('conjunction').join(arguments_list)
            if arguments:
                search_terms.append('({})'.format(arguments))

        # The 'retmax' parameter determines the maximum number of record ids
        # we want returned, where 100000 is the maximum. Setting it to maximum
        # will therefore return all available record ids.
        try:
            search_handle = Entrez.esearch(
                    db='pubmed',
                    term=' AND '.join(search_terms),
                    mindate=pub_date_start,
                    maxdate=pub_date_end,
                    datetype='pdat',
                    usehistory='y',
                    retmax='100000'
            )
        except Exception as e:
            warnings.warn(
                'An exception occurred ({0}).'.format(e),
                RuntimeWarning
            )
            if self.error_callback:
                self.error_callback(e)
            return

        # Read into a dictionary.
        search_results = Entrez.read(search_handle)
        search_handle.close()

        # Store data from this search.
        self.record_id_list = search_results.get('IdList')
        self.search_record_count = int(search_results.get('Count'))
        self.search_record_web_env = search_results.get('WebEnv')
        self.search_record_query_key = search_results.get('QueryKey')

    def _search_for_records_advanced(self, query):
        """Executes a advanced/custom search for PubMed records.

        The query is in the same format as the one assembled on the PubMed
        website. Results of the search, like QueryKey, WebEnv and Count are
        stored as parameters of this class's instance.

        Args:
            query (str): The advanced query string.
        """
        if not query:
            warnings.warn('Cannot run PubMed query on empty input.',
                          RuntimeWarning)
            return

        try:
            search_handle = Entrez.esearch(
                    db='pubmed',
                    term=query,
                    usehistory='y',
                    retmax='100000'
            )
        except Exception as e:
            warnings.warn(
                'An exception occurred ({0}).'.format(e),
                RuntimeWarning
            )
            if self.error_callback:
                self.error_callback(e)
            return
        search_results = Entrez.read(search_handle)
        search_handle.close()

        # Store data from this search.
        self.record_id_list = search_results.get('IdList')
        self.search_record_count = int(search_results.get('Count'))
        self.search_record_web_env = search_results.get('WebEnv')
        self.search_record_query_key = search_results.get('QueryKey')

    def _retrieve_record_batch(self, batch_start, batch_size):
        """Retrieves a PubMed article record batch.

        Retrieval is based on the info recovered by '_search_for_records()'.
        The batch size is limited by the 'batch_start' and 'batch_size'
        parameters. Returns a string containing the article info, if execution
        was successful and returns None otherwise.

        Args:
            batch_start (int): Specifies the starting index of this record
                batch.
            batch_size (int): Specifies the size of this records batch.

        Returns:
            list: A list of dictionaries that hold the data for each record.
        """
        if None in [self.search_record_web_env, self.search_record_query_key]:
            raise ValueError(  # Perform a search first!
                    'No WebEnv or QueryKey data in this PubMed class instance.'
            )

        fetch_handle = Entrez.efetch(
                db='pubmed',
                rettype='medline',
                retmode='text',
                retstart=batch_start,
                retmax=batch_size,
                webenv=self.search_record_web_env,
                query_key=self.search_record_query_key
        )

        data = list(Medline.parse(fetch_handle))
        fetch_handle.close()

        with shelve.open(self.cache_path) as query_cache:
            for record in data:
                id_key = record.get('PMID')
                if id_key is not None:
                    query_cache[id_key] = record
            query_cache.close()

        if self.progress_callback:
            self.progress_callback()

        return data

    def _retrieve_records(self, num_records,
                          includes_metadata=PUBMED_TEXT_FIELDS,
                          use_cache=True):
        """Retrieves the records queried with '_search_for_records()'.

        If retrieval was successful, generates a corpus with the text fields as
        meta attributes.

        Args:
            num_records (int): The number of records we wish to retrieve.

        Returns:
            `orangecontrib.text.corpus.Corpus`: The retrieved PubMed records
                as a corpus.
        """
        corpus = None
        batch_size = min(self.DEFAULT_BATCH_SIZE, num_records)
        cached_data = []  # Later on, construct the corpus from this.
        new_records = []  # Must download.

        if use_cache:
            with shelve.open(self.cache_path) as query_cache:
                # --- Retrieve cached ---
                # case 1: [all cached]
                # case 2: [all new]
                # case 3: [cached + new]
                for rec_id in self.record_id_list:
                    if (len(cached_data) + len(new_records)) == num_records:
                        break  # Capped.

                    record = query_cache.get(rec_id)
                    if record is not None:
                        cached_data.append(record)
                        continue

                    new_records.append(rec_id)
                query_cache.close()
        else:
            new_records = self.record_id_list

        cached_data_size = len(cached_data)
        if self.progress_callback:  # Advance the callback accordingly.
            for _ in itertools.repeat(None, int(cached_data_size/batch_size)):
                self.progress_callback()

        # --- Retrieve missing/new ---
        if len(new_records) > 0:
            if cached_data_size > 0:  # Some records were cached.
                # Create a starting corpus.
                corpus = _corpus_from_records(
                        cached_data,
                        includes_metadata
                )

            try:
                post_handle = Entrez.epost('pubmed', id=','.join(new_records))
                post_results = Entrez.read(post_handle)
                post_handle.close()
            except Exception as e:
                warnings.warn(
                    'An exception occurred ({0}).'.format(e),
                    RuntimeWarning
                )
                if self.error_callback:
                    self.error_callback(e)
                return

            self.search_record_web_env = post_results['WebEnv']
            self.search_record_query_key = post_results['QueryKey']

            # Fetch the records.
            for start in range(0, num_records, batch_size):
                try:
                    records = self._retrieve_record_batch(
                            start, batch_size
                    )
                except Exception as e:
                    warnings.warn(
                        'An exception occurred ({0}).'.format(e),
                        RuntimeWarning
                    )
                    if self.error_callback:
                        self.error_callback(e)
                    return
                meta_values, class_values = _records_to_corpus_entries(
                        records,
                        includes_metadata=includes_metadata
                )
                if corpus is None:
                    corpus = _corpus_from_records(
                            records,
                            includes_metadata
                    )
                else:  # Update the corpus.
                    corpus.extend_corpus(
                            meta_values,
                            class_values
                    )
            return corpus
        else:  # No new records, create a corpus from cached ones.
            return _corpus_from_records(
                    cached_data,
                    includes_metadata
            )

    def download_records(self, terms=[], authors=[],
                         pub_date_start=None, pub_date_end=None,
                         num_records=10, use_cache=True):
        """Downloads the requested records.

        Args:
            terms (list): Keywords to search by.
            authors (list): Authors to search by.
            pub_date_start (str): The start of the publish date chronological
                interval.
            pub_date_end (str): The end of the publish date chronological
                interval.
            num_records (int): The number of records we wish to retrieve.

        Returns:
            `orangecontrib.text.corpus.Corpus`: The constructed corpus.
        """
        self._search_for_records(terms, authors, pub_date_start, pub_date_end)
        return self._retrieve_records(num_records, use_cache=use_cache)