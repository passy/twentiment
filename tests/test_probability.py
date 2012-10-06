"""
Test for customizations of the thirdparty probability module.

:author: 2012, Pascal Hartig <phartig@weluse.de>
:license: Apache 2
"""

from unittest import TestCase
from twentiment.thirdparty.probability import _NINF


class ProbabilityTestCase(TestCase):

    def test_sum_logs_ninf(self):
        from twentiment.thirdparty.probability import sum_logs

        self.assertEqual(sum_logs([]), _NINF)
