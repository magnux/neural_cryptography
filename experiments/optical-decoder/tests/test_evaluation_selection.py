"""Coverage of bounded validation sets, independent of file ordering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import select_samples

class SelectionTests(unittest.TestCase):
    def sample(self, session, index):
        return SimpleNamespace(id=f'{session}-{index:05d}', metadata={
            'sessionId':session,'size':64,'colors':16,'source':'real'})

    def test_session_coverage_and_order_independence(self):
        samples=[self.sample('large',i) for i in range(100)] + [self.sample('small',i) for i in range(2)]
        chosen=select_samples(samples,6)
        self.assertEqual(sum(x.metadata['sessionId']=='small' for x in chosen),2)
        self.assertEqual([x.id for x in chosen],[x.id for x in select_samples(list(reversed(samples)),6)])
        self.assertEqual(len({x.id for x in chosen}),6)

    def test_all_samples_are_allowed_without_offline_count_cap(self):
        samples=[self.sample('a',i) for i in range(5000)]
        self.assertEqual(len(select_samples(samples,0)),5000)
        self.assertEqual(len(select_samples(samples,9000)),5000)
        self.assertEqual(len(select_samples(samples,4097)),4097)
        with self.assertRaises(ValueError):select_samples(samples,-1)

if __name__=='__main__':unittest.main()
