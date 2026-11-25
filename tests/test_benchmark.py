import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import benchmark as b

class BenchmarkTests(unittest.TestCase):
    def test_dataset(self):
        data = json.loads((b.ROOT / 'dataset.json').read_text())
        b.validate_dataset(data)
        self.assertEqual(len(data), 48)
        self.assertEqual({k: sum(d['label'] == k for d in data) for k in b.CRITERIA}, {k:12 for k in b.CRITERIA})

    def test_shared_rules_and_no_truth_leak(self):
        for m in b.MODELS:
            p = json.dumps(b.payload(m, 'hello'))
            self.assertIn('hello', p)
            self.assertNotIn('truth', p)
            for label in b.CRITERIA:
                self.assertIn(label, p)

    def test_decisions_response(self):
        response = {'answers': {'category': {'choice':'Invoice','confidence':.9,'probabilities':{'Invoice':.9,'Receipt':.05,'Contract':.03,'Other':.02}}}, 'usage':{'total_input_tokens':200,'total_output_tokens':0}}
        self.assertEqual(b.parse('decisions',response)[:2], ('Invoice',.9))
        response['answers']['category']['probabilities']['Other'] = .8
        with self.assertRaises(ValueError): b.parse('decisions',response)

    def test_generative_tokens_and_exact_label(self):
        r={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'Receipt'}]}}], 'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':2,'thoughtsTokenCount':7}}
        p=b.parse('generative',r)
        self.assertEqual(p[3:5],(100,9))
        r['candidates'][0]['content']['parts'][0]['text']='The answer is Receipt.'
        with self.assertRaises(ValueError): b.parse('generative',r)

    def test_metrics_errors_p95_missing_cost(self):
        rows=[{'method':'decisions','truth':'Invoice','prediction':'Invoice','error':None,'latency_ms':i,'estimated_usd':.001} for i in range(1,21)]
        rows.append(dict(rows[0], error='HTTP 429', prediction=None, latency_ms=999, estimated_usd=None))
        s=b.summarize(rows)[0]
        self.assertEqual(s['p95_latency_ms_valid'],19)
        self.assertAlmostEqual(s['accuracy_all'],20/21)
        self.assertAlmostEqual(s['error_rate'],1/21)
        self.assertIsNone(s['estimated_usd_per_1000_attempts'])
        with tempfile.TemporaryDirectory() as tmp:
            b.report(rows,Path(tmp))
            self.assertTrue((Path(tmp)/'benchmark.svg').exists())

    def test_transport_serialization(self):
        from io import BytesIO
        class Response(BytesIO):
            status = 200
        with patch.object(b.urllib.request, 'urlopen', return_value=Response(b'{"answers": {}}')) as call:
            status, response = b.post('decisions', {'input':'sample'}, 'test-only')
        req = call.call_args.args[0]
        self.assertEqual(status,200)
        self.assertEqual(req.full_url,b.BASE+'decisions')
        self.assertEqual(json.loads(req.data),{'input':'sample'})
        self.assertEqual(req.get_header('X-goog-api-key'),'test-only')

if __name__=='__main__': unittest.main()
