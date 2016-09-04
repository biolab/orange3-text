from datetime import datetime
import os
import csv
from newspaper import Article, ArticleException
from Orange.canvas.utils import environ
import warnings

ARTICLE_TEXT_FIELDS=["Article","Author","Date","Title","URL"]


cache_path = None
cache_file='article_data.csv'
cache_folder = os.path.join(environ.buffer_dir, "articlecache")
try:
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_path = os.path.join(cache_folder, cache_file)
except:
    warnings.warn('Could not assemble article query cache path', RuntimeWarning)


def _date_to_str(input_date):
    """
    Returns a string representation of the input date, according to the ISO 8601 format.
    :param input_date:
    :type input_date: datetime
    :return: str
    """
    iso=input_date.isoformat()
    date_part = iso.strip().split("T")[0].split("-")
    return "%s%s%s" % (date_part[0], date_part[1], date_part[2])

def _get_info(url):
    """
    Gets relevent available information from the url
    :param url:
    :type url: string
    :return : list
    """

    url=str(url)        # remove irrevelent variables
    is_cached=False
    if not os.path.exists(cache_path):
        with open (cache_path,'w') as query_cache:
            writer=csv.writer(query_cache,delimiter='\t')
            writer.writerow(ARTICLE_TEXT_FIELDS)
    with open (cache_path,'r') as query_cache:
        cache_data=csv.reader(query_cache,delimiter='\t')
        for row in cache_data:
            if (row[4]==url) :
                is_cached=True
                scraped_data=row  #no need to get the data again, if acraped details are already there in the cache
                break
    if not is_cached :          # if not cached, get the data
        article = Article(url)
        try:
            article.download()
            article.parse()
            article.nlp()
        except:
            pass
        scraped_data=[]
        text = article.text if article.text else ""
        text = ' '.join(text.split()) # remove white spaces
        title = article.title if article.title else ""
        title = ' '.join(title.split())  # remove white spaces
        authors = ', '.join(article.authors)
        authors = authors if authors else ""
        pub_date = _date_to_str(article.publish_date) if article.publish_date else ""
        web_url= article.url if article.url else ""
        scraped_data=[text, authors, pub_date, title, web_url]
        with open (cache_path, 'a') as query_cache:
            writer=csv.writer(query_cache,delimiter='\t')
            writer.writerow(scraped_data)
            is_cached=True
    return scraped_data, is_cached
