twentiment
==========

Research project on twitter sentiment analysis using the Naïve Bayes
Classificator.

Installation
------------

Install from PyPI (soon) or github with::

    pip install -e git+https://github.com:passy/twentiment.git

Usage
-----

First, start the twentiment server that loads the data from a JSON file. A
sample is available `in the repository <https://github.com/passy/twentiment/blob/623f4064469850b40b50db4707f12a07047f022b/samples/few_tweets.json>`_.

::

    twentiment_server samples/few_tweets.json

After that, you can use ``twentiment_client`` to query the server using the
syntax ``GUESS my tweet to be scored``.

Example
-------

::

    twentiment> GUESS hello world
    OK 0.0
    twentiment> GUESS This car is amazing.
    OK 0.5
    twentiment> GUESS My best friend is great.
    OK 0.9285714285714286
    twentiment> GUESS Whatever.
    OK 0.0
    twentiment> GUESS This car is horrible.
    OK -0.5
    twentiment> GUESS I am not looking forward to my appointment tomorrow.
    OK -0.9852941176470597


Wishlist
--------

(Ranked by importance)

    * Have a web-frontend that searches for tweets and rates their sentiment.
    * Give the server an option to fork the server process into the background
      and launch a shell like twentiment_client right away.
    * Restructure the Classifier to allow adaptive retraining, i.e. provide a
      TRAIN command that adds new samples at runtime.
        * At the moment, most of the calculations are done at start-up time, so
          querying is rather cheap. Could be difficult to find a good balance.

    * Persistence of the server state. Maybe through redis? Only important with
      TRAIN functionality.
    * Add some sort of parallelism to the server, so querying doesn't block.
    * Add a way of importing live training data from twitter (like from
      analysing emoticons)

Motivation
----------

This is a project report for the Business Intelligence course. To increase the
learning potential, I tried to reuse as little as possible from the excellent
`NLTK <http://nltk.org/>`_ project and reimplemented the relevant parts myself.
