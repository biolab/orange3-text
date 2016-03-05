import os
import nltk
from newspaper import news_pool, Source, Article
import tldextract 
import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter

chkr = SpellChecker("en_US",filters=[EmailFilter,URLFilter])
d = enchant.request_dict("en_US")


#newspaper
#find spelling mistakes
#sentiment analysis
#pos tagging
#twitter_API
	
url_to_clean=raw_input("Enter Url:")
article = Article(url_to_clean)
article.download()
article.parse()

try:
  html_string = ElementTree.tostring(article.clean_top_node)
except:
  html_string = "Error converting html to string."

article.nlp()



a = {
          'html': html_string, 
         'authors': str(', '.join(article.authors)), 
         'title': article.title,
         'text': article.text,
       #  'top_image': article.top_image,
         'videos': str(', '.join(article.movies)),
         'keywords': str(', '.join(article.keywords)),
         'summary': article.summary
         }
def spell_check(check_text):
  chkr.set_text(check_text)
  old_err=[]
  new_spell=[]
  combine=[]
  for mistakes in chkr:
    old_err.append(mistakes.word)
    j=d.suggest(err.word)
    new_spell.append(j[1])
  combine.append(new_spell)
  combine.append(old_err)
  return combine

def pos_tag(tag_text):
  tokens=nltk.word_tokenize(tag_text)
  return nltk.pos_tag(tokens)
