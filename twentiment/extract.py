"""
Utils for extracting and preparing information.

:author: 2012, Pascal Hartig <phartig@weluse.de>
:license: Apache 2
"""

def extract_features(document):
    """Extract features from a document and returns them in a Bag Of Words
    model dict.
    """

    return {word: True for word in document}
