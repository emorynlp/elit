import unittest
import elit
import elit.pretrained.amr


class TestAMR(unittest.TestCase):
    def test_parse(self):
        parser = elit.load(elit.pretrained.amr.AMR3_BART_LARGE_EN)
        amr = parser('The boy wants the girl to believe him.')
        print(amr)


if __name__ == '__main__':
    unittest.main()
