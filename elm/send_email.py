"""Send an email via macOS Mail.app (AppleScript). Body is read from a file to
avoid shell/AppleScript quoting issues.

Usage:
    /usr/bin/python3 -m elm.send_email <to> <subject> <body_file>
    # validate without sending (creates then deletes a message):
    /usr/bin/python3 -m elm.send_email --dry-run
"""

from __future__ import annotations

import subprocess
import sys


def _osascript(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True, timeout=timeout)


def dry_run() -> bool:
    """Create an outgoing message then delete it — verifies the create path
    works without sending or leaving a draft."""
    script = '''
    tell application "Mail"
        set m to make new outgoing message with properties {subject:"__elm_dryrun__", content:"x", visible:false}
        delete m
    end tell
    return "ok"
    '''
    r = _osascript(script, timeout=30)
    print("dry-run rc=", r.returncode, "out=", r.stdout.strip(), "err=", r.stderr.strip())
    return r.returncode == 0


def send(to: str, subject: str, body_path: str) -> bool:
    # AppleScript reads the body file as UTF-8; subject is kept simple ASCII.
    subj = subject.replace('"', "'")
    script = f'''
    set theBody to (read POSIX file "{body_path}" as «class utf8»)
    tell application "Mail"
        set m to make new outgoing message with properties {{subject:"{subj}", content:theBody, visible:false}}
        tell m
            make new to recipient at end of to recipients with properties {{address:"{to}"}}
        end tell
        send m
    end tell
    return "sent"
    '''
    r = _osascript(script, timeout=90)
    ok = r.returncode == 0
    print(f"send rc={r.returncode} out={r.stdout.strip()} err={r.stderr.strip()}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--dry-run":
        sys.exit(0 if dry_run() else 1)
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(2)
    _to, _subj, _body = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.exit(0 if send(_to, _subj, _body) else 1)
