"""Parse ShapeHeadInfoFemale (decompiled) into data tables the deform consumes:
  - enums.json: DstName / SrcName ordered name lists (index -> cf_J_ / cf_s_ bone name)
  - update_eqns.json: the hardcoded src->dst assignments from Update()

Run after decompiling:  ilspycmd Assembly-CSharp.dll -t ShapeHeadInfoFemale > ShapeHeadInfoFemale.cs
    .venv/Scripts/python.exe scripts/hs2_parse_update.py <ShapeHeadInfoFemale.cs>
Outputs into data/hs2_head/. Idempotent; re-run if the game updates.
"""
import json
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_head")

FIELD = {"vctPos": "pos", "vctRot": "rot", "vctScl": "scl"}
AXIS = {"x": 0, "y": 1, "z": 2}


def parse_enum(src, name):
    m = re.search(r"enum\s+" + name + r"\s*\{(.*?)\}", src, re.S)
    body = m.group(1)
    names = [x.strip() for x in body.split(",")]
    return [x for x in names if x and not x.startswith("//")]


def parse_term(expr):
    """'dictSrc[41].vctPos.x' -> ('src',41,'pos',0); '1f'/'0f' -> ('const',v)."""
    expr = expr.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?f?", expr):
        return ["const", float(expr.rstrip("f"))]
    m = re.fullmatch(r"dictSrc\[(\d+)\]\.(vctPos|vctRot|vctScl)\.([xyz])", expr)
    if not m:
        raise ValueError(f"unparseable term: {expr!r}")
    return ["src", int(m.group(1)), FIELD[m.group(2)], AXIS[m.group(3)]]


def parse_arg(arg):
    """Sum of terms -> list of terms (added together)."""
    return [parse_term(t) for t in arg.split("+")]


def main():
    cs = sys.argv[1]
    src = open(cs, encoding="utf-8", errors="replace").read()
    dst_names = parse_enum(src, "DstName")
    src_names = parse_enum(src, "SrcName")

    upd = re.search(r"public override void Update\(\)\s*\{(.*?)\n\t\}", src, re.S).group(1)
    eqns = []  # each: {dst:int, target:'pos'|'rot'|'scl', axis:int|None, args:[...]}
    for line in upd.splitlines():
        line = line.strip()
        m = re.search(r"dictDst\[(\d+)\]\.trfBone\.SetLocal(\w+)\((.*)\);", line)
        if not m:
            continue
        dst = int(m.group(1)); call = m.group(2); argstr = m.group(3)
        if call in ("PositionX", "PositionY", "PositionZ"):
            axis = AXIS[call[-1].lower()]
            eqns.append({"dst": dst, "target": "pos", "axis": axis, "args": [parse_arg(argstr)]})
        elif call in ("Rotation", "Scale"):
            tgt = "rot" if call == "Rotation" else "scl"
            # 3 comma-separated args (x,y,z); split at top level (no nested parens here)
            parts = [p for p in argstr.split(",")]
            assert len(parts) == 3, f"{call} expects 3 args: {argstr}"
            eqns.append({"dst": dst, "target": tgt, "axis": None,
                         "args": [parse_arg(p) for p in parts]})
        else:
            raise ValueError(f"unhandled call SetLocal{call}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"dst": dst_names, "src": src_names},
              open(os.path.join(OUT, "enums.json"), "w"), indent=0)
    json.dump(eqns, open(os.path.join(OUT, "update_eqns.json"), "w"))
    print(f"DstName: {len(dst_names)}  SrcName: {len(src_names)}")
    print(f"Update assignments: {len(eqns)}")
    # quick integrity: all src/dst indices in range
    maxdst = max(e["dst"] for e in eqns)
    maxsrc = max(t[1] for e in eqns for a in e["args"] for t in a if t[0] == "src")
    print(f"max dst idx={maxdst} (<{len(dst_names)})  max src idx={maxsrc} (<{len(src_names)})")


if __name__ == "__main__":
    main()
