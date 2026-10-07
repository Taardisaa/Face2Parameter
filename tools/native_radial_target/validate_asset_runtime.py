"""NEW explicit asset-provider runtime revision; frozen validator remains unchanged."""
from __future__ import annotations

import argparse
import json

from tools.native_radial_target.asset_provider import prepare_asset_provider


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','contract','asset-contract','asset-contract-sha256','out-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--source-gaze-policy',choices=('reject','frozen_source_locals'),default='reject')
    a=p.parse_args()
    provider=prepare_asset_provider(a.asset_contract,a.asset_contract_sha256)
    report=provider.runtime_audit(a.manifest,a.contract,a.out_dir,source_gaze_policy=a.source_gaze_policy)
    print(json.dumps({'passed':report['passed'],'rejection':report.get('rejection'),'explicit_revision':report['explicit_asset_provider_revision']}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
