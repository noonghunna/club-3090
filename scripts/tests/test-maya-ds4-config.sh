#!/usr/bin/env bash
# Guard: the DeepSeek-V4.1-Flash Project Maya recipe (models/deepseek-v4.1-flash-1017/strata-maya/launcher/) writes
# club-3090's launch settings into maya-ds4.json at every boot (ds4_prepare.py config() through ds4_settings.py),
# and prepares the pack the way the slug needs. Runs the launcher's own functions against scratch folders (no
# /opt/maya, no weights):
#   1. no settings: effort high, sampling 1.0 / 1.0, name deepseek-v4.1-flash, maya-ds4 with --tp 1 (a layer
#      split on two GPUs), --vision off, --dspark on and a --cpu-threads from threads();
#   2. each setting lands on Maya's key; medium / xhigh become high / max; ENABLE_THINKING=false wins;
#      THREADS sets --cpu-threads; KV / SPEC_N write nothing (maya-ds4 has neither); LOOP_GUARD true / false
#      writes loop_guard (unset writes nothing: Maya's own, on);
#   3. the config is rewritten from scratch each time, so a setting removed is gone;
#   4. a bad value exits 2 from `validate` and stops ds4_prepare.py's main() before it looks at any file;
#   5. threads(): half the CPUs (nproc / 2), at least 1, and with no argument the same count as the shell's nproc;
#   6. verify(): a file whose SHA-256 is wrong stops the boot; a matching set is recorded and not hashed again;
#   7. every key a setting writes has evidence that the pinned Maya reads it (engine_config_reads.py check).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # the launcher folder is mounted into the container: leave no __pycache__
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
L="models/deepseek-v4.1-flash-1017/strata-maya/launcher"

rc=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; rc=1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
SETTINGS="REASONING_EFFORT ENABLE_THINKING THINKING_BUDGET MAX_TOKENS KV SPEC_N TEMP TOP_P TOP_K MIN_P PRESENCE_PENALTY REPEAT_PENALTY SERVED_NAME THREADS LOOP_GUARD"
clean() { for v in $SETTINGS; do printf -- '-u %s ' "$v"; done; }
# config() for ista-iq2-xxs into $work/ista-iq2-xxs/maya-ds4.json, with the given settings
write() {
  env $(clean) "$@" python3 - "$L" "$work" <<'PY'
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ds4_prepare
d = Path(sys.argv[2]) / "ista-iq2-xxs"; d.mkdir(exist_ok=True)
ds4_prepare.config("ista-iq2-xxs", d, d / "pack", types.SimpleNamespace(gpus="0,1", tp=1, context=65536))
PY
}
get() { python3 -c "import json,sys; c=json.load(open('$work/ista-iq2-xxs/maya-ds4.json')); a=c['args']; print(eval(sys.argv[1]))" "$1"; }
arg() { get "a[a.index('$1')+1] if '$1' in a else None"; }

echo "== 1. no settings"
write >/dev/null || bad "config() failed with no settings"
[[ "$(get 'c["reasoning_effort"]')" == high ]] && ok "effort high (Maya's own for DeepSeek)" || bad "effort: $(get 'c.get("reasoning_effort")')"
[[ "$(get 'c["sampling"]')" == "{'temperature': 1.0, 'top_p': 1.0}" ]] && ok "sampling 1.0 / 1.0" || bad "sampling: $(get 'c["sampling"]')"
[[ "$(get 'c["model_name"]')" == deepseek-v4.1-flash ]] && ok "served as deepseek-v4.1-flash" || bad "model_name: $(get 'c.get("model_name")')"
[[ "$(get 'c["exe"]')" == /opt/maya/build/maya-ds4 ]] && ok "exe maya-ds4" || bad "exe: $(get 'c["exe"]')"
# (Maya's model_label reads ds4_quant: without it, the files are named "EXL3 2.9 bpw" - Maya's own quant)
[[ "$(get 'c["installer"]["family"], c["installer"]["ds4_quant"]')" == "('deepseek', 'ISTA IQ2_XXS')" ]] \
  && ok "installer record: family deepseek, ds4_quant ISTA IQ2_XXS" || bad "installer: $(get 'c["installer"]')"
[[ "$(arg --tp) $(arg --vision) $(arg --dspark) $(arg --max-context)" == "1 off on 65536" ]] \
  && ok "--tp 1 (layer split), --vision off, --dspark on, --max-context 65536" || bad "args: $(get 'a')"
want="$(env $(clean) python3 -c "import sys; sys.path.insert(0, '$L'); import ds4_settings; print(ds4_settings.threads())")"
[[ "$(arg --cpu-threads)" == "$want" && "$want" -ge 1 ]] && ok "--cpu-threads $want = threads()" || bad "--cpu-threads $(arg --cpu-threads), threads() $want"
[[ "$(get '"thinking_budget" in c or "--kv" in a or "--spec" in a or "loop_guard" in c')" == False ]] \
  && ok "no thinking_budget / --kv / --spec / loop_guard: Maya's own" || bad "an unset setting was written"
[[ "$(get 'c["gpu"]') $(get 'c["log"].endswith("/ista-iq2-xxs/maya-ds4.log")')" == "[0, 1] True" ]] \
  && ok "GPUs [0, 1], the engine log in the quant's folder" || bad "gpu/log: $(get 'c["gpu"], c["log"]')"

echo "== 2. each setting reaches Maya's key"
write REASONING_EFFORT=medium THINKING_BUDGET=8192 TEMP=0.7 TOP_P=0.9 TOP_K=40 MIN_P=0.05 PRESENCE_PENALTY=0.5 \
      REPEAT_PENALTY=1.1 THREADS=7 KV=int8 SPEC_N=3 MAX_TOKENS=4096 >/dev/null || bad "config() failed"
[[ "$(get 'c["reasoning_effort"]')" == high ]] && ok "medium -> high" || bad "effort: $(get 'c["reasoning_effort"]')"
[[ "$(get 'c["thinking_budget"]')" == 8192 ]] && ok "thinking_budget" || bad "budget: $(get 'c.get("thinking_budget")')"
[[ "$(arg --cpu-threads)" == 7 ]] && ok "THREADS=7 -> --cpu-threads 7" || bad "threads: $(get 'a')"
[[ "$(get '"--kv" in a or "--spec" in a or "max_tokens" in c')" == False ]] \
  && ok "KV / SPEC_N / MAX_TOKENS write nothing (maya-ds4 has no --kv or depth flag)" || bad "KV/SPEC_N/MAX_TOKENS written: $(get 'a')"
[[ "$(get 'sorted(c["sampling"].items())')" == "[('min_p', 0.05), ('presence_penalty', 0.5), ('repetition_penalty', 1.1), ('temperature', 0.7), ('top_k', 40), ('top_p', 0.9)]" ]] \
  && ok "sampler settings" || bad "sampling: $(get 'c["sampling"]')"
write REASONING_EFFORT=xhigh >/dev/null; [[ "$(get 'c["reasoning_effort"]')" == max ]] && ok "xhigh -> max" || bad "xhigh"
write SERVED_NAME=llm >/dev/null; [[ "$(get 'c["model_name"]')" == llm ]] && ok "SERVED_NAME=llm -> model_name llm" || bad "served name: $(get 'c["model_name"]')"
write LOOP_GUARD=false >/dev/null; [[ "$(get 'c["loop_guard"]')" == False ]] && ok "LOOP_GUARD=false -> loop_guard false" || bad "loop guard off: $(get 'c.get("loop_guard")')"
write LOOP_GUARD=true >/dev/null; [[ "$(get 'c["loop_guard"]')" == True ]] && ok "LOOP_GUARD=true -> loop_guard true" || bad "loop guard on: $(get 'c.get("loop_guard")')"
out="$(write ENABLE_THINKING=false REASONING_EFFORT=low)"
[[ "$(get 'c["reasoning_effort"]')" == none ]] && command grep -q "ENABLE_THINKING=false wins" <<<"$out" \
  && ok "ENABLE_THINKING=false -> none, with a note" || bad "thinking off: $out"

echo "== 3. a removed setting is gone"
write >/dev/null
[[ "$(get 'c["reasoning_effort"], "thinking_budget" in c, c["model_name"], a[a.index("--cpu-threads")+1], "loop_guard" in c')" == "('high', False, 'deepseek-v4.1-flash', '$want', False)" ]] \
  && ok "back to the defaults" || bad "a setting lingered: $(get 'c["reasoning_effort"], "thinking_budget" in c, c["model_name"], a')"

echo "== 4. a bad value"
for probe in REASONING_EFFORT=minimal REASONING_EFFORT=75 ENABLE_THINKING=off THINKING_BUDGET=-1 TEMP=x TOP_P=0 \
             TOP_K=65 MIN_P=2 PRESENCE_PENALTY=-1 REPEAT_PENALTY=0 THREADS=0 THREADS=four LOOP_GUARD=off LOOP_GUARD=0; do
  err="$(env $(clean) "$probe" python3 "$L/ds4_settings.py" validate 2>&1)"; r=$?
  [[ $r -eq 2 && "$err" == *"${probe%%=*}"* ]] && ok "$probe refused" || bad "$probe: rc=$r $err"
done
err="$(env $(clean) "SERVED_NAME=my model" python3 "$L/ds4_settings.py" validate 2>&1)"; r=$?
[[ $r -eq 2 && "$err" == *SERVED_NAME* ]] && ok "SERVED_NAME with a space refused" || bad "SERVED_NAME=my model: rc=$r $err"
err="$(env $(clean) REASONING_EFFORT=minimal python3 "$L/ds4_prepare.py" --tag ista-iq2-xxs --src "$work/nowhere" \
       --dspark "$work/nowhere" --data "$work" 2>&1)"; r=$?
[[ $r -ne 0 && "$err" == *REASONING_EFFORT* && "$err" != *missing* ]] \
  && ok "ds4_prepare.py stops on it before looking at the files" || bad "main(): rc=$r $err"
err="$(env $(clean) python3 "$L/ds4_prepare.py" --tag ista-iq2-xxs --src "$work/nowhere" --dspark "$work/nowhere" \
       --data "$work" 2>&1)"; r=$?
[[ $r -ne 0 && "$err" == *missing:* && "$err" == *setup.sh* ]] \
  && ok "missing weights: stops, names the files and the download" || bad "missing weights: rc=$r $err"

echo "== 5. threads(): half the CPUs (nproc / 2), at least 1"
out="$(python3 - "$L" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import ds4_settings as s
print(s.threads(list(range(32))), s.threads(list(range(16))), s.threads(list(range(5))), s.threads([0, 1]),
      s.threads([0]), s.threads([]))
PY
)"
# 32 -> 16 (the old 3/4 rule gave 24 here); 16 -> 8; odd counts round down; 1 CPU and none still get 1
[[ "$out" == "16 8 2 1 1 1" ]] && ok "32 -> 16, 16 -> 8, 5 -> 2, 2 -> 1, 1 -> 1, 0 -> 1" || bad "threads(): $out"
# with no argument: this process's CPUs, the count nproc prints (nproc honours OMP_NUM_THREADS, so clear it)
mine="$(python3 -c "import sys; sys.path.insert(0, '$L'); import ds4_settings; print(ds4_settings.threads())")"
shell="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"; shell=$(( shell / 2 )); (( shell < 1 )) && shell=1
[[ "$mine" == "$shell" ]] && ok "threads() = nproc / 2 here ($mine)" || bad "threads() $mine, nproc / 2 $shell"

echo "== 6. verify(): the SHA-256 table stops a damaged file; a match is recorded"
out="$(python3 - "$L" "$work" <<'PY' 2>&1
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ds4_prepare as p
w = Path(sys.argv[2]) / "v"; w.mkdir()
f = w / "a.gguf"; f.write_bytes(b"good")
p.WEIGHTS["t"] = {"gguf": {"a.gguf": hashlib.sha256(b"good").hexdigest()}, "dspark": {}}
p.verify("t", [f], w)                               # matches: recorded
calls = []
p.sha256 = lambda x: calls.append(x) or "never"     # a second verify must not hash again
p.verify("t", [f], w)
print("rehashed" if calls else "not rehashed")
f.write_bytes(b"bad!")                               # same size, different bytes ...
st = f.stat()                                        # ... and a later mtime, set: two writes inside one tick of
os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # the kernel's file clock share one (2.3% of runs, 2026-10-29)
p.sha256 = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
p.verify("t", [f], w)
print("NOT STOPPED")
PY
)"
command grep -q "every file matches" <<<"$out" && command grep -q "not rehashed" <<<"$out" \
  && ok "a matching file is recorded and not hashed again" || bad "verify record: $out"
command grep -q "SHA-256 mismatch" <<<"$out" && ! command grep -q "NOT STOPPED" <<<"$out" \
  && ok "a changed file is hashed again and stops the boot" || bad "verify mismatch: $out"

echo "== 7. the pinned engine reads every key a setting writes (engine-config-reads.json; no docker)"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/lib/profiles/engine_config_reads.py check --engine strata-maya:ds4 || rc=1

echo "== 8. the per-tag files: no drafter for ud-iq3-s, maya-xs's separate Engram set, each tag's files = the model profile's"
out="$(env $(clean) python3 - "$L" "$work" <<'PY' 2>&1
import json, re, subprocess, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ds4_prepare as p
w = Path(sys.argv[2])
for tag in ("ista-iq2-xxs", "ud-iq3-s", "maya-xs"):
    d = w / ("c-" + tag); d.mkdir()
    p.config(tag, d, d / "pack", types.SimpleNamespace(gpus="0,1", tp=1, context=65536))
    c = json.loads((d / "maya-ds4.json").read_text())
    a = c["args"]
    print(f"{tag} dspark={a[a.index('--dspark') + 1]} ds4_quant={c['installer']['ds4_quant']}")
# files_of(): the 8 shards present and neither a /dspark nor an /engram folder at all is enough for
# ud-iq3-s - its Engram tables are inside its own shards, so it has no "engram" key
src = w / "UD-IQ3_S"; src.mkdir()
for n in p.WEIGHTS["ud-iq3-s"]["gguf"]:
    (src / n).write_bytes(b"x")
model, tables, drafter = p.files_of("ud-iq3-s", src, w / "no-dspark-folder", w / "no-engram-folder")
print(f"files_of ud-iq3-s: {len(model)} model files, {len(tables)} table files, {len(drafter)} drafter files")
# pack(): the argv pack_gguf.py gets, without running it
seen = []
def fake_run(argv, cwd=None):
    seen.append(argv)
    out = Path(argv[argv.index("--out") + 1]); out.mkdir(parents=True)
    (out / "tensors.txt").write_text("")
    return types.SimpleNamespace(returncode=0)
p.subprocess = types.SimpleNamespace(run=fake_run)
p.pack(model, tables, drafter, w / "p-ud", False)
dj = w / "dspark-xs2.json"; dj.write_text("{}"); db = w / "dspark-xs2.bin"; db.write_bytes(b"x")
(w / "p-ista").mkdir()
p.pack(model, tables, [db, dj], w / "p-ista", False)
print("pack ud-iq3-s drafter arg:", "--drafter" in seen[0])
print("pack with a drafter:", seen[1][seen[1].index("--drafter") + 1] == str(dj) if "--drafter" in seen[1] else False)
print("pack with no Engram set passes no --engram:", "--engram" not in seen[0])
# maya-xs: its Q6_K tables are a GGUF set of their own mounted at /engram, and the pack has to be
# told where they are - pack_gguf.py stops at "the shards of more than one GGUF split" on a folder
# that holds both, which is why the compose mounts the two files rather than the folder.
xs = w / "Maya-XS"; xs.mkdir(); eg = w / "engram"; eg.mkdir(); dsp = w / "dspark"; dsp.mkdir()
for n in p.WEIGHTS["maya-xs"]["gguf"]:
    (xs / n).write_bytes(b"x")
for n in p.WEIGHTS["maya-xs"]["engram"]:
    (eg / n).write_bytes(b"x")
for n in p.WEIGHTS["maya-xs"]["dspark"]:
    (dsp / n).write_bytes(b"x")
xm, xt, xd = p.files_of("maya-xs", xs, dsp, eg)
(w / "p-xs").mkdir()
p.pack(xm, xt, xd, w / "p-xs", False)
print(f"files_of maya-xs: {len(xm)} model files, {len(xt)} table files, {len(xd)} drafter files")
print("pack maya-xs --engram is the table folder:",
      seen[2][seen[2].index("--engram") + 1] == str(eg) if "--engram" in seen[2] else False)
print("pack maya-xs --shards names only the model's own:",
      seen[2][seen[2].index("--shards") + 1:] == [f.name for f in xm])
# the SHA-256 table: 64 hex digits each, and the names the model profile's files: list downloads
bad = [n for n, h in p.WEIGHTS["ud-iq3-s"]["gguf"].items() if not re.fullmatch(r"[0-9a-f]{64}", h)]
prof = Path("scripts/lib/profiles/models/deepseek-v4.1-flash-1017.yml").read_text(encoding="utf-8")
listed = sorted(m.group(1) for m in re.finditer(r"^\s+- UD-IQ3_S/(\S+\.gguf)$", prof, re.M))
print("sha table ok:", not bad, "| profile files == launcher table:", listed == sorted(p.WEIGHTS["ud-iq3-s"]["gguf"]))
xs_bad = [n for k in ("gguf", "engram", "dspark") for n, h in p.WEIGHTS["maya-xs"][k].items()
          if not re.fullmatch(r"[0-9a-f]{64}", h)]
xs_listed = sorted(m.group(1) for m in re.finditer(r"^\s+- Maya-XS/(\S+\.gguf)$", prof, re.M))
print("maya-xs sha table ok:", not xs_bad, "| profile files == launcher gguf+engram:",
      xs_listed == sorted(list(p.WEIGHTS["maya-xs"]["gguf"]) + list(p.WEIGHTS["maya-xs"]["engram"])))
PY
)"
command grep -q "^ista-iq2-xxs dspark=on ds4_quant=ISTA IQ2_XXS$" <<<"$out" && command grep -q "^ud-iq3-s dspark=off ds4_quant=gguf-iq3_s$" <<<"$out" \
  && ok "ista-iq2-xxs: --dspark on; ud-iq3-s: --dspark off, ds4_quant gguf-iq3_s (Maya shows UD-IQ3_S)" || bad "config per tag: $out"
command grep -q "^maya-xs dspark=on ds4_quant=maya-xs$" <<<"$out" \
  && ok "maya-xs: --dspark on, ds4_quant maya-xs (one of Maya's own quants, so Maya names it)" || bad "maya-xs config: $out"
command grep -q "files_of ud-iq3-s: 8 model files, 0 table files, 0 drafter files" <<<"$out" \
  && ok "files_of(): 8 shards, no drafter folder and no Engram folder needed" || bad "files_of: $out"
command grep -q "pack ud-iq3-s drafter arg: False" <<<"$out" && command grep -q "pack with a drafter: True" <<<"$out" \
  && ok "pack(): no --drafter without a drafter; the drafter's .json passed with one" || bad "pack argv: $out"
command grep -q "pack with no Engram set passes no --engram: True" <<<"$out" \
  && ok "pack(): a tag whose tables are among its shards passes no --engram" || bad "pack engram argv: $out"
command grep -q "files_of maya-xs: 3 model files, 2 table files, 2 drafter files" <<<"$out" \
  && ok "files_of(): maya-xs reads 3 shards + the 2-file Engram set + the drafter pair" || bad "files_of maya-xs: $out"
command grep -q "pack maya-xs --engram is the table folder: True" <<<"$out" \
  && command grep -q "pack maya-xs --shards names only the model's own: True" <<<"$out" \
  && ok "pack(): maya-xs passes --engram <table folder> and --shards without the table files" || bad "pack maya-xs argv: $out"
command grep -q "sha table ok: True | profile files == launcher table: True" <<<"$out" \
  && ok "ud-iq3-s: 8 SHA-256s, the same files the model profile downloads" || bad "sha table: $out"
command grep -q "maya-xs sha table ok: True | profile files == launcher gguf+engram: True" <<<"$out" \
  && ok "maya-xs: 7 SHA-256s (Maya's own), the same files the model profile downloads" || bad "maya-xs sha table: $out"

if [[ "$rc" -eq 0 ]]; then echo "test-maya-ds4-config: ok"; else echo "test-maya-ds4-config: FAILED" >&2; fi
exit "$rc"
