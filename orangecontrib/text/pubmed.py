import os
import shelve
import warnings
from datetime import datetime

import numpy as np
from Bio import Entrez
from Bio import Medline
from validate_email import validate_email

from Orange.canvas.utils import environ
from Orange.data import StringVariable, DiscreteVariable, TimeVariable, Domain
from orangecontrib.text.corpus import Corpus

BASE_ENTRY_URL = 'http://www.ncbi.nlm.nih.gov/pubmed/?term='

PUBMED_FIELD_AUTHORS = 'authors'
PUBMED_FIELD_TITLE = 'article_title'
PUBMED_FIELD_HEADINGS = 'mesh_headings'
PUBMED_FIELD_ABSTRACT = 'abstract'
PUBMED_FIELD_URL = 'url'
PUBMED_FIELD_DATE = 'pub_date'
PUBMED_TEXT_FIELDS = [
    (PUBMED_FIELD_AUTHORS, 'FAU'),
    (PUBMED_FIELD_TITLE, 'TI'),
    (PUBMED_FIELD_HEADINGS, 'MH'),
    (PUBMED_FIELD_ABSTRACT, 'AB'),
    (PUBMED_FIELD_URL, 'PMID'),
    (PUBMED_FIELD_DATE, 'DP'),
]


def _mesh_headings_to_class(mesh_headings):
    """Extract a class from the mesh headings.

    We take the first mesh heading and extract it to use as a class.

    Args:
        mesh_headings (list): The list containing all the mesh headings.

    Returns:
        str: The class value.
    """
    # e.g. heading1/heading2,heading3/*heading4
    return mesh_headings[0].split('/')[0].split(' ')[0].replace('*', '').lower()


def _date_to_iso(date):
    possible_date_formats = [
        '%Y %b %d',
        '%Y %b',
        '%Y',
    ]

    season_mapping = {
        'fall': 'Sep',
        'autumn': 'Sep',
        'winter': 'Dec',
        'spring': 'Mar',
        'summer': 'Jun',
    }

    date = date.lower()
    # Seasons to their respective months.
    for season, month in season_mapping.items():
        date = date.replace(season, month)
    date = date.split('-')[0]  # 2015 Sep-Dec --> 2015 Sep

    time_var = TimeVariable()
    for date_format in possible_date_formats:
        try:
            date_string = datetime.strptime(
                    date, date_format
            ).date().isoformat()
            return time_var.parse(date_string)
        except ValueError:
            continue  # Try the next format.

    warnings.warn(
            'Could not parse "{}" into a date.'.format(date),
            RuntimeWarning
    )
    return time_var.parse(np.nan)


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
        fields = []

        for field_name, field_key in includes_metadata:
            field_value = record.get(field_key, '')

            if isinstance(field_value, list):
                fields.append(' '.join(field_value))
            elif field_name == PUBMED_FIELD_URL:
                fields.append('{}{}'.format(BASE_ENTRY_URL, field_value))
            elif field_name == PUBMED_FIELD_DATE:
                fields.append(_date_to_iso(field_value))
            else:
                fields.append(field_value)

        metadata[row_num] = np.array(fields, dtype=object)[None, :]

        mesh_headings = record.get('MH', record.get('OT'))
        if mesh_headings is not None:
            mesh_headings = _mesh_headings_to_class(mesh_headings)
        class_values.append(mesh_headings)

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
    meta_vars = []
    for field_name, _ in includes_metadata:
        if field_name == 'pub_date':
            meta_vars.append(TimeVariable(field_name))
        else:
            meta_vars.append(StringVariable.make(field_name))

    class_vars = [
        DiscreteVariable('section_name', values=list(set(class_values)))
    ]
    domain = Domain([], class_vars=class_vars, metas=meta_vars)

    Y = np.array([class_vars[0].to_val(cv) for cv in class_values])[:, None]

    return Corpus(domain=domain, Y=Y, metas=meta_values)


class Pubmed:
    MAX_RECORDS = 100000
    MAX_BATCH_SIZE = 1000

    def __init__(self, email, progress_callback=None, error_callback=None):
        if not validate_email(email):
            raise ValueError('{} is not a valid email address.'.format(email))

        Entrez.email = email

        self.record_id_list = None  # Ids of the records available.
        self.search_record_count = 0  # Number of records.
        self.search_record_web_env = None  # BioPython history param.
        self.search_record_query_key = None  # BioPython history param.

        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.stop_signal = False

        self.cache_path = None
        cache_folder = os.path.join(environ.buffer_dir, 'pubmedcache')

        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        self.cache_path = os.path.join(cache_folder, 'query_cache')

    def _search_for_records(self, terms=[], authors=[],
                            pub_date_start=None, pub_date_end=None,
                            advanced_query=None):
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

        query_dict = {
            'db': 'pubmed',
            'usehistory': 'y',
            'retmax': self.MAX_RECORDS
        }

        if advanced_query is not None:
            query_dict.update({'term': advanced_query})
        else:
            search_terms = []
            argument_checks = [
                (authors, 'Author', ' AND ',
                 '{} is not a valid data type for the field "authors".'.format(
                         type(authors)
                 )),
                (terms, 'All fields', ' OR ',
                 '{} is not a valid data type for the field "terms".'.format(
                         type(terms)
                 )),
            ]

            for argument, field, conjunction, error in argument_checks:
                if not isinstance(argument, list):
                    raise ValueError(error)
                arguments_list = [
                    '{}[{}]'.format(arg, field) for arg in argument
                    ]
                arguments = conjunction.join(arguments_list)
                if arguments:
                    search_terms.append('({})'.format(arguments))

            query_dict.update({
                'term': ' AND '.join(search_terms),
                'mindate': pub_date_start.replace('-', '/'),
                'maxdate': pub_date_end.replace('-', '/'),
                'datetype': 'pdat',
            })

        # The 'retmax' parameter determines the maximum number of record ids
        # we want returned, where 100000 is the maximum. Setting it to maximum
        # will therefore return all available record ids.
        try:
            search_handle = Entrez.esearch(**query_dict)
        except IOError as e:
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

        data = Medline.parse(fetch_handle)
        records = [record for record in data]
        fetch_handle.close()

        return records

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
        batch_size = min(self.MAX_BATCH_SIZE, num_records)
        cached_data = []  # Later on, construct the corpus from this.
        new_records = []  # Must download.

        # if use_cache:
        with shelve.open(self.cache_path) as query_cache:
            # --- Retrieve cached ---
            # case 1: [all cached]
            # case 2: [all new]
            # case 3: [cached + new]
            for rec_id in self.record_id_list:
                if (len(cached_data) + len(new_records)) == num_records:
                    break  # Capped.

                if use_cache:
                    record = query_cache.get(rec_id)
                    if record is not None:
                        cached_data.append(record)
                    else:
                        new_records.append(rec_id)
                else:
                    new_records.append(rec_id)
            query_cache.close()

        cached_data_size = len(cached_data)
        if cached_data_size > 0:
            # Advance the callback accordingly.
            self.progress_callback(int(cached_data_size/batch_size))

            # Create a starting corpus.
            corpus = _corpus_from_records(cached_data, includes_metadata)

        # --- Retrieve missing/new ---
        if len(new_records) > 0:
            try:
                post_handle = Entrez.epost('pubmed', id=','.join(new_records))
                post_results = Entrez.read(post_handle)
                post_handle.close()
            except IOError as e:
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
            for start in range(0, len(new_records), batch_size):
                if self.stop_signal:
                    self.stop_signal = False
                    break

                try:
                    records = self._retrieve_record_batch(start, batch_size)
                except Exception as e:
                    warnings.warn(
                        'An exception occurred ({0}).'.format(e),
                        RuntimeWarning
                    )
                    if self.error_callback:
                        self.error_callback(e)
                    return

                # Add the entries to shelve.
                with shelve.open(self.cache_path) as query_cache:
                    for record in records:
                        id_key = record.get('PMID')
                        if id_key is not None:
                            query_cache[id_key] = record
                    query_cache.close()

                # Advance progress.
                if self.progress_callback:
                    self.progress_callback()

                meta_values, class_values = _records_to_corpus_entries(
                        records,
                        includes_metadata=includes_metadata
                )

                if corpus is None:
                    corpus = _corpus_from_records(records, includes_metadata)
                else:  # Update the corpus.
                    corpus.extend_corpus(meta_values, class_values)

        return corpus

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

    def stop_retrieving(self):
        self.stop_signal = True
