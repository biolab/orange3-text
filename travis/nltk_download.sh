#!/usr/bin/env bash
if [ "$(ls -A $HOME/nltk_data)" ]; then
    echo "Using cached NLTK data containing folders:";
    ls $HOME/nltk_data;
else
    python -m nltk.downloader wordnet;
fi
