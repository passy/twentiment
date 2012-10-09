"""
Tests for the text utils.
"""


import doctest
from unittest import TestCase


class TextUtilsTestCase(TestCase):

    def test_doctests(self):
        from twentiment import text

        doctest.testmod(text)
