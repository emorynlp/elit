import unittest
import elit


class TestAMR(unittest.TestCase):
    def test_parse(self):
        parser = elit.load('KOREAN_TREEBANK_BIAFFINE_DEP')
        tree = parser('그는 르노가 3 월말까지 인수제의 시한을 갖고 있다고 덧붙였다 .'.split())
        print(tree)


if __name__ == '__main__':
    unittest.main()
