import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('nlp', Path(__file__).resolve().parents[1] / 'server' / 'nlp_service.py')
nlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nlp)

class NlpTests(unittest.TestCase):
    def test_basic_and_underthesea(self):
        self.assertEqual([t['word'] for t in nlp.tokenize_rows(['Học sinh!'], 'basic')[0]], ['học', 'sinh'])
        row = nlp.tokenize_rows(['Học sinh học tập.'], 'underthesea')[0]
        self.assertTrue(any(t['word'] == 'học_sinh' and t['pos'].startswith('N') for t in row))

    def test_text_binary_and_bad_file(self):
        original = nlp.MODELS
        with tempfile.TemporaryDirectory() as folder:
            nlp.MODELS = Path(folder)
            try:
                (nlp.MODELS / 'fixture.vec').write_text('3 2\nhọc_sinh 1 0\nhọc 0 1\n\u00a0 .2 .3\n', encoding='utf-8')
                self.assertEqual(nlp.load_model('fixture.vec')['dim'], 2)
                data = nlp.handle({'action':'prepare','lines':['học_sinh học'],'embedding':True,'tokenizer':'basic'})
                self.assertEqual(data['vectors']['học_sinh'], [1, 0])
                (nlp.MODELS / 'fixture.bin').write_bytes(b'1 2\nword ' + struct.pack('<ff', .5, -.5) + b'\n')
                self.assertEqual(nlp.load_model('fixture.bin')['count'], 1)
                (nlp.MODELS / 'bad.vec').write_text('1 2\nword nan 0\n')
                with self.assertRaises(ValueError):
                    nlp.load_model('bad.vec')
                with self.assertRaises(ValueError):
                    nlp.load_model('../secret.txt')
            finally:
                if nlp.MODEL:
                    nlp.MODEL['db'].close()
                nlp.MODEL = None
                nlp.MODELS = original

if __name__ == '__main__':
    unittest.main()
