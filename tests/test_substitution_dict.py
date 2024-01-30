"""!@package ep_bolfi.tests.test_substitution_dict
Test suite for ep_bolfi.utility.preprocessing.SubstitutionDict.
"""


import unittest

from ep_bolfi.utility.preprocessing import SubstitutionDict


class TestSubstitutionDict(unittest.TestCase):
    """!@brief
    Test suite for ep_bolfi.utility.preprocessing.SubstitutionDict.
    """

    def initialize_test_dict(self):

        def func(passed_dict):
            return passed_dict['dolor'] + passed_dict['sit']

        base_dict = {
            "lorem": 1,
            "ipsum": 2.0,
        }
        substitutions = {
            "dolor": 'ipsum',
            "sit": lambda passed_dict: passed_dict['lorem'],
            "amet": func,
        }
        expected_result = {
            "lorem": 1,
            "ipsum": 2.0,
            "dolor": 2.0,
            "sit": 1,
            "amet": 3.0,
        }
        return (SubstitutionDict(base_dict, substitutions), expected_result)

    def test_getitem(self):
        test_dict, expected_result = self.initialize_test_dict()
        for k, v in test_dict.items():
            self.assertEqual(v, expected_result[k])

    def test_change(self):
        test_dict, _ = self.initialize_test_dict()
        test_dict['lorem'] = 2
        test_dict['ipsum'] = 4.0
        self.assertEqual(test_dict['dolor'], 4.0)
        self.assertEqual(test_dict['sit'], 2)
        self.assertEqual(test_dict['amet'], 6.0)


if __name__ == '__main__':
    unittest.main()
