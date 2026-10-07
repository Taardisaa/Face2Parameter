"""NEW strict asset revision CLI; no frozen V1/source/cache edits."""
from __future__ import annotations

import argparse
import json

from tools.native_radial_target.asset_provider_strict import (
    prepare_strict_asset_provider,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','contract','asset-contract','asset-contract-sha256','out-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--source-gaze-policy',choices=('reject','frozen_source_locals'),default='reject')
    a=p.parse_args();provider=prepare_strict_asset_provider(a.asset_contract,a.asset_contract_sha256)
    result=provider.runtime_audit(a.manifest,a.contract,a.out_dir,source_gaze_policy=a.source_gaze_policy)
    print(json.dumps({'passed':result['passed'],'rejection':result.get('rejection'),'strict_asset_provider':result['strict_asset_provider']}))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
