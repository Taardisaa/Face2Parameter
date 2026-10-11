"""Compare complete installed/source/patched ilspy IL; preserve non-capture types."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def normalized(text):
    # Only PE placement labels/comments; never erase instructions or blob bytes.
    text = re.sub(r'\bI_[0-9A-F]+\b', 'DATA_RVA', text)
    return re.sub(r'// Method begins at RVA 0x[0-9a-fA-F]+',
                  '// Method begins at RVA RELOCATED', text)


def types(text):
    starts = list(re.finditer(r'^\.class ', text, re.M))
    result = {}
    for i, start in enumerate(starts):
        block = text[start.start():starts[i+1].start() if i+1 < len(starts) else len(text)]
        endings = re.findall(r'\} // end of class ([^\r\n]+)', block)
        if not endings or endings[-1] in result:
            raise ValueError('Incomplete or duplicate top-level IL type')
        result[endings[-1]] = normalized(block)
    if not result:
        raise ValueError('No IL types; refusal to certify an empty comparison')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('installed_il', 'baseline_il', 'patched_il', 'installed_dll', 'patched_dll', 'out'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    text = {k: getattr(a, k).read_text(encoding='utf-8-sig')
            for k in ('installed_il', 'baseline_il', 'patched_il')}
    base_equal = normalized(text['installed_il']) == normalized(text['baseline_il'])
    old, new = types(text['installed_il']), types(text['patched_il'])
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    expected = {'HS2Mod.McpBridge.MakerRenderService', 'HS2Mod.McpBridge.MakerRenderStyleService'}
    report = dict(baseline_all_il_equal_except_rva=base_equal, top_level_type_count=len(old),
                  patched_changed_types=changed, non_capture_types_preserved=set(changed) <= expected,
                  inputs={k: {'path': str(getattr(a, k).resolve()),
                              'sha256': hashlib.sha256(getattr(a, k).read_bytes()).hexdigest()}
                          for k in ('installed_il', 'baseline_il', 'patched_il', 'installed_dll', 'patched_dll')},
                  scope='All emitted ilspy IL text, signatures, attributes and data bytes. Only method RVA comments and initialized-data RVA placement labels normalized; no unrelated IL ignored.')
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))
    if not base_equal or set(changed) != expected:
        raise SystemExit('Installed baseline/patch preservation failed; do not deploy.')


if __name__ == '__main__':
    main()
