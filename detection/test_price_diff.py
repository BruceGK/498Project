import unittest

from detection.price_diff import PriceMeasure


class TestPriceMeasure(unittest.TestCase):

    def test_load_successful(self):
        measure = PriceMeasure('rmsc03_two_hour_final', 'impact', 'no_impact')
        self.assertTrue(len(measure.fundamental) == len(measure.impact_data))
        self.assertTrue(len(measure.fundamental) == len(measure.no_impact_data))
        self.assertTrue(len(measure.fundamental) > 0)

    def test_self_one(self):
        measure = PriceMeasure('rmsc03_two_hour_final', 'impact', 'impact')
        result = measure.compare()
        self.assertEqual(int(result.item()), 1)


if __name__ == '__main__':
    unittest.main()
