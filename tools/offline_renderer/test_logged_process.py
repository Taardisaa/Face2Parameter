"""CPU-only regression for output preservation; no native DLL or graphics."""
import tempfile
import sys
from pathlib import Path
import unittest
from .logged_process import run_logged

class LoggedProcessTests(unittest.TestCase):
    def test_utf8_output_survives_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'timeout'
            r=run_logged([sys.executable,'-c',"import sys,time; sys.stdout.buffer.write('中文 ready\\n'.encode()); sys.stdout.flush(); sys.stderr.buffer.write(b'error before wait\\n'); sys.stderr.flush(); time.sleep(10)"],out,timeout=.8)
            self.assertEqual(r['status'],'timeout')
            self.assertEqual((out/'stdout.bin').read_bytes(),'中文 ready\n'.encode())
            self.assertEqual((out/'stderr.bin').read_bytes(),b'error before wait\n')
            self.assertTrue((out/'receipt.json').exists())
    def test_nonzero_retains_both_streams(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'failed'
            r=run_logged([sys.executable,'-c',"import sys; print('progress'); print('failed',file=sys.stderr); sys.exit(7)"],out,timeout=5)
            self.assertEqual(r['status'],'failed_exit');self.assertEqual(r['returncode'],7)
            self.assertIn(b'progress',(out/'stdout.bin').read_bytes());self.assertIn(b'failed',(out/'stderr.bin').read_bytes())
    def test_launch_error_retained(self):
        with tempfile.TemporaryDirectory() as d:
            r=run_logged(['missing_fixture_that_does_not_exist'],Path(d)/'launch_error',timeout=1)
            self.assertEqual(r['status'],'launch_failed');self.assertIn('error',r)
if __name__=='__main__':unittest.main()
