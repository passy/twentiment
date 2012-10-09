#!/usr/bin/env python3
# coding: utf-8
"""
Utils for text parsing, manipulation and normalization.

:author: 2012, Pascal Hartig <phartig@weluse.de>
:license: Apache 2
"""

import re
import string


EMOTICONS = frozenset([':)', ':(', '):', '(:', '(-:', ')-:', ':-)', ':-('])


def normalize_text(text):
    """Formats text to strip unneccesary words, punctuation and whitespace.
    Returns a tokenized list.

    :param text: Text to process.

    >>> text = "ommmmmmg how'r U!? VISI T  <html> <a href='http://google.com'> my</a> site @ http://www.coolstuff.com haha"
    >>> normalize_text(text)
    ['ommg', 'howr', 'visi', 'my', 'site', 'haha']

    >>> normalize_text("FOE JAPAN が粘り強く主張していた避難の権利")
    ['foe', 'japan', '\u304c\u7c98\u308a\u5f37\u304f\u4e3b\u5f35\u3057\u3066\u3044\u305f\u907f\u96e3\u306e\u6a29\u5229']

    >>> normalize_text('no ')
    ['no']

    >>> normalize_text('')
    []
    """

    if not text:
        return []

    # Process in lower case
    text = text.lower()

    patterns = (
        ("@[A-Za-z0-9_]+", ''),     # mentions
        ("#[A-Za-z0-9_]+", ''),     # hash tags
        (r"(\w)\1{2,}", r"\1\1"),   # occurences of more than two consecutive
                                    # repeating characters
        ("<[^<]+?>", ''),           # sgml tags
        ("(http|www)[^ ]*", ''),    # URLs
    )

    for pattern in patterns:
        text = re.sub(pattern[0], pattern[1], text)

    # Temporarily store emoticons before stripping.
    emoticons = set([e for e in EMOTICONS if e in text])

    # Remove puctuation, leading/trailing whitespace
    text = text.translate({ord(x): None for x in string.punctuation}).strip()

    # Reappand emoticons
    text += ' '.join([e for e in emoticons])

    return [w for w in re.split(r'\s', text) if len(w) > 1]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
