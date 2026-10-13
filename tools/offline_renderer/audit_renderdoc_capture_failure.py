"""Read pinned RenderDoc D3D11 log evidence; never starts or controls a process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE_COMMIT = "e4bd23b671d3d5a747ff5221dbe08a63eb6ca200"
SUMMARY = re.compile(r"Failed to capture, frame (\d+): (.+?)\s*$")
COMMAND = "Don't have command list "
UNMAP = "Saw an Unmap that we didn't capture the corresponding Map for"


def classify(text: str) -> dict:
    """Preserve attempts separately: later warnings must not diagnose an older failure.

    Only the driver's final failure summary establishes its selected failure enum.
    It can overwrite an earlier cause, so preserve both observed warning branches.
    Partial/buffered logs or a warning alone do not establish the failed attempt.
    """
    attempts = []
    current = None
    orphan_events = []
    for number, line in enumerate(text.splitlines(), 1):
        event = None
        if "Starting capture" in line:
            if current is not None:
                attempts.append(current)
            current = {"start_line": number, "status": "incomplete_log", "events": []}
        elif COMMAND in line and "This frame is unsuccessful." in line:
            event = {"line": number, "branch": "uncaptured_command_list", "text": line}
        elif UNMAP in line:
            event = {"line": number, "branch": "uncaptured_map_unmap", "text": line}
        elif match := SUMMARY.search(line):
            reason = match.group(2)
            branch = {"Uncapped command list": "uncaptured_command_list",
                      "Uncapped Map()/Unmap()": "uncaptured_map_unmap"}.get(reason, "unknown")
            event = {"line": number, "branch": branch, "text": line, "frame": int(match.group(1))}
            if current is not None:
                current.update(status="driver_rejected", selected_reason=branch,
                               final_reason_string=reason, end_line=number)
        elif "Finished capture, Frame " in line:
            event = {"line": number, "branch": "driver_finished", "text": line}
            if current is not None:
                current.update(status="driver_finished_not_file_verified", end_line=number)
        if event is not None:
            if current is None:
                orphan_events.append(event)
            else:
                current["events"].append(event)
                if "end_line" in current:
                    attempts.append(current)
                    current = None
    if current is not None:
        attempts.append(current)
    return {"attempts": attempts, "orphan_events": orphan_events,
            "native_capture_accepted": False,
            "scope": "Log evidence only; no graphics, file replay or capture success assertion",
            "source_commit": SOURCE_COMMIT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = args.log.read_bytes()
    # Retain malformed log bytes in the original hashed input; replacement only
    # affects the human-readable line evidence, never the file provenance.
    result = classify(data.decode("utf-8-sig", errors="replace"))
    result["input"] = {"path": str(args.log.resolve()), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({"attempts": len(result["attempts"]), "receipt": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
