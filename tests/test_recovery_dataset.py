import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery import RecoveryPlanner
from tools.dataset_pipeline import prepare, validate_label
from tools.preflight import check

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self, name, content):
        d = self.root/'raw'/name
        d.mkdir(parents=True)
        (d/'frame.png').write_bytes(content)
        (d/'frame.txt').write_text('0 0.5 0.5 0.2 0.4')

    def test_invalid_labels(self):
        p = self.root/'bad.txt'
        for value in ['0 nan .5 .2 .2','-1 .5 .5 .2 .2','0 .01 .5 .5 .5','0 .5 .5 0 .5','0.5 .5 .5 .2 .2','0 .5 .5 .2']:
            with self.subTest(value=value):
                p.write_text(value)
                with self.assertRaises(ValueError): validate_label(p,2)

    def test_background(self):
        p=self.root/'empty.txt'
        p.write_text('')
        validate_label(p,2)

    def test_split_and_refuse_stale_output(self):
        self.session('one', b'one')
        self.session('two', b'two')
        args=(self.root/'raw',self.root/'images',self.root/'out',['enemy','teammate'])
        report=prepare(*args)
        self.assertEqual(report['counts'], {'train':1,'valid':1})
        train={p.stem for p in (self.root/'out/train/images').iterdir()}
        valid={p.stem for p in (self.root/'out/valid/images').iterdir()}
        self.assertFalse(train & valid)
        with self.assertRaises(ValueError): prepare(*args)

    def test_duplicates_not_in_both_splits(self):
        self.session('one',b'same')
        self.session('two',b'same')
        with self.assertRaises(ValueError):
            prepare(self.root/'raw',self.root/'images',self.root/'out',['enemy'])

    def test_missing_assets_reported(self):
        (self.root/'config.json').write_text('{}')
        self.assertEqual(len(check('config.json',self.root)),4)

    def test_escalation_and_reset(self):
        r=RecoveryPlanner()
        first=r.next(0,100,1)
        second=r.next(4,100,1)
        self.assertNotEqual(first[0],second[0])
        self.assertGreater(second[1], first[1])
        self.assertEqual(first,r.next(30,100,1))
        for i in range(100): self.assertLessEqual(r.next(31+i,100,1)[1],3)

if __name__ == '__main__': unittest.main()
