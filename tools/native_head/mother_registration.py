"""Register a complete authored bundle as an independent native HS2 head/skin.

Copies the audited installed head2 list contracts. Materials, textures and
animation references are inherited intact, not rebuilt by the legacy builder.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file


def csv_bytes(category, row, name):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    for value in ([category], [0], [name], list(row), list(row.values())):
        writer.writerow(value)
    return stream.getvalue().encode('utf-8-sig')


def package(bundle_dir, audit_path, out, title):
    if out.exists():
        raise FileExistsError('Use a fresh package directory')
    bundle = json.loads((bundle_dir/'receipt.json').read_text())
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    donor = next(r for r in audit['native'] if r['id']==2)
    if sha(donor['bundle']) != bundle['native_donor']['sha256']:
        raise ValueError('List contract and complete asset donor differ')
    if sha(bundle['bundle']['path']) != bundle['bundle']['sha256']:
        raise ValueError('Authored bundle changed')
    key = bundle['asset_key']
    guid = 'codex.native.'+key
    head = {k:v for k,v in donor['row'].items() if not k.startswith('_')}
    head.update(ID='1', Name=title, MainManifest='abdata', MainAB=bundle['main_ab'],
                MainData=bundle['prefab'], Preset='')
    skin = {k:v for k,v in donor['compatible_skin_rows'][0].items() if not k.startswith('_')}
    skin.update(ID='1', HeadID='1', Name=title+' 原生皮肤')
    root = ET.Element('manifest', {'schema-ver':'1'})
    for k,v in dict(guid=guid, name=title, version='0.1.0', author='Codex',
            description='Independent native-topology development head. Original eight parts, expression tables, UV and native skin assets retained. Requires installed head2 dependencies.').items():
        ET.SubElement(root,k).text=v
    ET.SubElement(root,'faceSkinInfo', {'skinID':'1','headID':'1','headGUID':guid})
    out.mkdir(parents=True)
    path = out/(key+'.zipmod')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.xml', ET.tostring(root,encoding='utf-8',xml_declaration=True))
        archive.write(bundle['bundle']['path'], 'abdata/'+bundle['main_ab'])
        archive.writestr('abdata/list/characustom/codex_'+key+'_fo_head_00.csv',
                         csv_bytes(210,head,'codex_'+key+'_fo_head_00'))
        archive.writestr('abdata/list/characustom/codex_'+key+'_ft_skin_f_00.csv',
                         csv_bytes(211,skin,'codex_'+key+'_ft_skin_f_00'))
    with zipfile.ZipFile(path) as archive:
        if archive.read('abdata/'+bundle['main_ab']) != Path(bundle['bundle']['path']).read_bytes():
            raise ValueError('Packaged bundle differs')
        if ET.fromstring(archive.read('manifest.xml')).findtext('guid') != guid:
            raise ValueError('Independent identity differs')
    save_json(out/'receipt.json',dict(format='native_mother_registration_v1',
        package=source_file(path), source_bundle=source_file(bundle_dir/'receipt.json'),
        list_audit=source_file(audit_path), code=source_file(Path(__file__)),
        guid=guid, original_head_slot=1, original_skin_slot=1, head_row=head, skin_row=skin,
        independent_registration=True, original_skin_textures_unmodified=True,
        installed=False, game_loading_verified=False, full_animation_compatibility_certified=False))
    print(json.dumps(dict(package=str(path.resolve()),guid=guid)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=Path('outputs/head_base_audit_20261007/prefab_contract_v1.json'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--title',required=True)
    a=p.parse_args()
    package(a.bundle.resolve(),a.audit.resolve(),a.out.resolve(),a.title)
