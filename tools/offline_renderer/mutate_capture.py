"""Apply explicitly witnessed exact data patches to official XML+ZIP, CPU only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.render.capture_mutation import mutate_capture
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',required=True,type=Path);ap.add_argument('--plan',required=True,type=Path)
    ap.add_argument('--out-xml',required=True,type=Path);ap.add_argument('--receipt',required=True,type=Path)
    a=ap.parse_args()
    if a.receipt.exists():raise FileExistsError('Fresh receipt required')
    result=mutate_capture(a.xml,a.plan,a.out_xml)
    a.receipt.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
