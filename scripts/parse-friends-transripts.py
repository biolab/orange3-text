"""
For this script to run you need a ZIP from http://home.versatel.nl/friendspic0102/
as well as all ten pages season01.html - season10.html with transcripts info.
"""
import os
import re
import logging
from bs4 import BeautifulSoup

PATH = './friendsalltranscripts'


def unify_name(name):
    """
    Unify author names.

    Args:
        name: Author's name to fix

    Returns: Fixed name
    """
    fix = {
        '': '',
        'r Zelner': 'Mr Zelner',
        'Professor Sherman': 'Prof. Sherman',
        'Dr Green': 'Dr. Green',
        'DR HORTON': 'DR. HORTON',
        'MRS GREEN': 'MRS. GREEN',
        'FBOB': 'Fun Bobby',
        'ESTL': 'Estelle',
        'RTST': 'Mr. Ratstatter',
        'RAHCEL': 'RACHEL',
        'RACH': 'RACHEL',
        'MNCA': 'MONICA',
        'MICH': 'MICHAEL',
        'CHAN': 'CHANDLER',
        'PHOE': 'PHOEBE',
    }
    if name == 'MR, GELLER':
        name = 'MR. GELLER'

    # split by ',', '/' , ...
    # eg. CHAN,MNCA
    # eg. CHAN/MNCA
    authors = [i.strip() for i in re.split(',|/|and |AND |&', name) if i.strip()]
    for i, a in enumerate(authors):
        if a in fix:
            authors[i] = fix[a]
    name = ', '.join(authors)
    return name.title()     # CamelCase


def season_episode_number(name):
    """
    Extract Season & Episode number from file name.

    Args:
        name: File name

    Returns: Season number & episode number
    """
    # fix inconsistent file names for season & episode parsing
    fix = {
        '0423uncut.html': '0423-0424.html',
        '0624.html': '0624-0625.html',
        '0723.html': '0723-0724.html',
        '0823.html': '0823-0824.html',
    }
    if name in fix:
        name = fix[name]

    # get season & episode number
    num = name.split('.')[0]
    season = num[:2]
    episode = num[2:4]
    if '-' in num:
        episode += '-' + num[7:9]
    return season, episode


#
# Load info from html files 'season01.html' - 'season10.html'
#
episodes = []
for season in range(1, 11):
    file = 'season{:02d}.html'.format(season)
    content = open(os.path.join(PATH, file), 'r')
    content = ''.join(content.readlines())
    soup = BeautifulSoup(content, 'lxml')

    for i in soup.find('tbody').find_all('tr', {'class': ['evenrow', 'unevenrow']}):
        table_cells = i.find_all('td')

        url = table_cells[1].find('a')['href'].split('/')[-1]
        title = table_cells[3].text.strip()
        date = table_cells[4].text

        # fix two part titles
        if '\n' in title:
            title = title.split('Part I')[0] + 'Part I & 2'

        # skip Episode on Oprah and Outtakes
        if 'oprah' not in url and 'outtakes' not in url:
            episodes.append((url, title, date))

#
# Parse and save transcripts into .tab file
#
with open('friends-transcripts.tab', 'w') as f:
    # Write header
    f.write('\t'.join(['Season', 'Episode', 'Season & Episode', 'Title', 'Author', 'Quote']) + '\n')
    f.write('\t'.join(['d', 'd', 'd', 'd', 'd', 'string']) + '\n')
    f.write('\t'.join(['meta', 'meta', 'meta', 'meta', 'meta', 'include=True']) + '\n')
    f.write('\n')

    # Parse transcripts
    for file, title, date in episodes:
        print('Working on {}: {}'.format(file, title))

        # Read the transcript HTML
        content = open(os.path.join(PATH, file), 'r', encoding='ISO-8859-1')
        content = ''.join(content.readlines())
        content = content.replace('<br>', '</p><p>')    # replace breaks with paragraphs
        soup = BeautifulSoup(content, 'lxml')

        # Get season &episode number
        season, episode = season_episode_number(file)

        # Extract all quotes for this episode
        for par in soup.findAll('p'):
            line = par.text.replace('\n', '')

            if any(line.startswith(i) for i in ['[', '(']):
                logging.warning("SKIPPED: {}".format(line))
                continue

            # delete everything between [] and () and {}
            line = re.sub(r'\([^)]*\)', '', line)
            line = re.sub(r'\[[^)]*\]', '', line)
            line = re.sub(r'\{[^)]*\}', '', line)

            # Skip lines not in format:
            # person: quote
            if ':' not in line:
                logging.warning("SKIPPED: {}".format(line))
                continue

            # get author and quote
            author, quote = line.split(':', 1)
            author = unify_name(author)

            # ignore some authors
            IGNORE = ['Written', 'Transcri', 'Adjustments', 'With Help From', 'Story by', 'Teleplay by',
                      'Note', 'Directed', 'CUT TO', 'Russian to Roman alphabet', 'Aired']
            if any(i.lower() in author.lower() for i in IGNORE):
                logging.warning("SKIPPED: {}".format(line))
                continue

            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(season, episode, season+episode, title, author, quote))
