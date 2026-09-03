#!/usr/bin/env bash
# Permission hygiene for scripts/ — a shebang means "run me", so the file must be
# executable. Drift is silent: nothing fails until someone types ./scripts/x.sh
# and gets "Permission denied", and the whole suite invokes tests as `bash <path>`
# so the exec bit is never exercised there.
#
# Snapshot when this landed: 209 shebang files under scripts/, 41 not executable.
# Nothing was BROKEN (the docs invoke via `bash …`, and the one true ./ invocation
# — ./scripts/switch.sh — was already +x), but `./scripts/report.sh` and
# `./scripts/setup.sh` both returned Permission denied, which is not what a reader
# expects of a script carrying `#!/usr/bin/env bash`.
#
# SOURCED_OK below are files that are `source`d or imported rather than executed.
# A shebang there is decorative and +x buys nothing, so they are exempt BY NAME —
# adding one is a conscious decision, not a silent exemption.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import subprocess, sys

# Sourced / imported, never executed — a shebang is decorative here.
SOURCED_OK = {
    "scripts/lib/bench-row-formatter.sh",   # sourced by submit-bench.sh
    "scripts/lib/capture.sh",               # sourced by bench.sh
    "scripts/lib/card.sh",                  # sourced by bench.sh
    "scripts/lib/compose-meta.sh",          # sourced by preflight.sh
    "scripts/lib/gpu-select.sh",            # sourced by launch.sh
    "scripts/lib/p2p-state.sh",             # sourced by preflight.sh
    "scripts/lib/report_calib.sh",          # sourced by test-report-calib.sh
    "scripts/lib/profiles/repo_dotenv.py",  # imported by promote.py / export_pr.py
    "scripts/lib/profiles/weights.py",      # imported by the profile loaders
    "scripts/tests/fixtures/report-harness/report-env.sh",  # test fixture, sourced
    "scripts/tests/fixtures/soak-harness/soak-env.sh",      # test fixture, sourced
}

out = subprocess.run(["git", "ls-files", "-s", "scripts/"],
                     capture_output=True, text=True).stdout
missing, stray = [], []
for line in out.splitlines():
    mode, _, _, path = line.split(maxsplit=3)
    if not path.endswith((".sh", ".py")):
        continue
    try:
        has_shebang = open(path, "rb").read(2) == b"#!"
    except OSError:
        continue
    if has_shebang and mode == "100644" and path not in SOURCED_OK:
        missing.append(path)
    if not has_shebang and mode == "100755":
        stray.append(path)

if missing or stray:
    print("test-script-permissions: FAIL\n")
    for p in missing:
        print(f"  {p}")
        print(f"    has a shebang but is not executable in git (100644).")
        print(f"    FIX: chmod +x {p}   — or add it to SOURCED_OK in this test if it "
              f"is sourced/imported, never executed.\n")
    for p in stray:
        print(f"  {p}")
        print(f"    is executable (100755) but has NO shebang — it cannot be run "
              f"directly.\n    FIX: add a shebang, or chmod -x {p}.\n")
    sys.exit(1)

print("test-script-permissions: PASS — every shebang script under scripts/ is "
      "executable (sourced/imported libraries exempted by name)")
PY
