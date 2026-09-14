import os
from pathlib import Path
import sys
import tempfile
import unittest
from bench_speed_tradeoff import Worker, summarize

class TradeoffTests(unittest.TestCase):
    def rows(self):
        return [{'mode':'text-only','steps':8,'variant':v,'pair':i,
                 'run':{'elapsed_seconds':t,'peak_rss_kib':1024,'minor_faults':0,
                        'major_faults':0,'encode_seconds':1,'sample_seconds':2,'decode_seconds':1},
                 'audio_seconds':1,'wav_sha256':'same','idle_warm':{'passed':True},
                 'idle_run':{'passed':True},'swap_delta':{'pswpin':0,'pswpout':0}}
                for i in range(1,6) for v,t in [('baseline',10),('candidate',8)]]
    def test_acceptance_requires_all_conditions(self):
        rows=self.rows()
        self.assertTrue(summarize(rows,5,False)['text-only-s8']['latency_candidate_gate'])
        self.assertFalse(summarize(rows,5,True)['text-only-s8']['latency_candidate_gate'])
        for field in ['busy','swap','parity','regression']:
            rows=self.rows()
            if field=='busy':rows[0]['idle_run']['passed']=False
            if field=='swap':rows[0]['swap_delta']['pswpin']=1
            if field=='parity':rows[0]['wav_sha256']='different'
            if field=='regression':rows[1]['run']['elapsed_seconds']=11
            self.assertFalse(summarize(rows,5,False)['text-only-s8']['latency_candidate_gate'])
        self.assertEqual(summarize(self.rows()[:-1],5,False),{})
    def test_multiple_buffered_lines_and_startup_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Worker([sys.executable,'-c','print("log"); print(\'__IRO_READY__={"ok":true}\'); print(\'__IRO_RESULT__={"done":true}\')'],os.environ.copy(),Path(tmp)/'ok.log',2)
            try:
                self.assertTrue(w.until('__IRO_READY__=')['ok'])
                self.assertTrue(w.until('__IRO_RESULT__=')['done'])
            finally:w.close()
            w=Worker([sys.executable,'-c','raise SystemExit(2)'],os.environ.copy(),Path(tmp)/'bad.log',2)
            try:
                with self.assertRaises(RuntimeError):w.until('__IRO_READY__=')
            finally:w.close()
    def test_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Worker([sys.executable,'-c','import time;time.sleep(30)'],os.environ.copy(),Path(tmp)/'slow.log',.05)
            try:
                with self.assertRaises(RuntimeError):w.until('__IRO_READY__=')
            finally:
                w.p.terminate();w.close()
            self.assertIsNotNone(w.p.returncode)

if __name__=='__main__':unittest.main()
