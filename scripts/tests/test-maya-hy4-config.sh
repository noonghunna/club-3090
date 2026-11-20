#!/usr/bin/env bash
# Guard: the Hy4-Flash Project Maya recipe (models/hy4-flash/strata-maya/launcher/) writes club-3090's launch settings
# into maya-hy4.json at every boot (hy4_prepare.py config() through hy4_settings.py), and prepares the pack the way the
# slugs need. Runs the launcher's own functions against scratch folders (no /opt/maya, no weights):
#   1. no settings: effort high, sampling 0.9 / 1.0, name hy4-flash, maya-hy4, --mtp and a --cpu-threads from threads();
#      no --kv, no --vision, no --profile (v1.1.68 ships no Hy4 expert profile), no env; the installer record Maya's
#      model_label reads (family hy4, hy4_quant maya-l);
#   2. each setting lands on Maya's key: REASONING_EFFORT none is thinking off, any other level runs at high (Hy4-Flash's
#      two levels) with a note; ENABLE_THINKING=false wins; KV, DRAFTING, VISION and THREADS set their flags;
#      HY4_EG_POPULATE goes into the config's env; LOOP_GUARD writes loop_guard; SPEC_N / MAX_TOKENS write nothing;
#   3. the config is rewritten from scratch each time, so a setting removed is gone;
#   4. a bad value exits 2 from `validate` and stops hy4_prepare.py's main() before it looks at any file; missing weights
#      stop the boot naming the files and the download;
#   5. threads(): half the CPUs, at least 1;
#   6. pack(): pack_gguf.py gets the two shards by name, the vision tower and the MTP block, and --out on the data volume;
#   7. verify(): a file whose SHA-256 is wrong stops the boot; a matching set is recorded and not hashed again;
#   8. every key a setting writes has evidence that the pinned Maya reads it (engine_config_reads.py check).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # the launcher folder is mounted into the container: leave no __pycache__
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
L="models/hy4-flash/strata-maya/launcher"

rc=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; rc=1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
SETTINGS="REASONING_EFFORT ENABLE_THINKING THINKING_BUDGET MAX_TOKENS KV SPEC_N TEMP TOP_P TOP_K MIN_P PRESENCE_PENALTY REPEAT_PENALTY SERVED_NAME THREADS LOOP_GUARD DRAFTING VISION HY4_EG_POPULATE CONTEXT"
clean() { for v in $SETTINGS; do printf -- '-u %s ' "$v"; done; }
# config() for hy4-maya-l into $work/hy4-maya-l/maya-hy4.json, with the given settings
write() {
  env $(clean) "$@" python3 - "$L" "$work" <<'PY'
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hy4_prepare
d = Path(sys.argv[2]) / "hy4-maya-l"; d.mkdir(exist_ok=True)
hy4_prepare.config("hy4-maya-l", d, d / "pack", types.SimpleNamespace(gpus="0,1", context=32768))
PY
}
get() { python3 -c "import json,sys; c=json.load(open('$work/hy4-maya-l/maya-hy4.json')); a=c['args']; print(eval(sys.argv[1]))" "$1"; }
arg() { get "a[a.index('$1')+1] if '$1' in a else None"; }
has() { get "'$1' in a"; }

echo "== 1. no settings"
write >/dev/null || bad "config() failed with no settings"
[[ "$(get 'c["reasoning_effort"]')" == high ]] && ok "effort high (Hy4-Flash's own, as Maya's setup writes it)" || bad "effort: $(get 'c.get("reasoning_effort")')"
[[ "$(get 'c["sampling"]')" == "{'temperature': 0.9, 'top_p': 1.0}" ]] && ok "sampling 0.9 / 1.0 (Tencent's)" || bad "sampling: $(get 'c["sampling"]')"
[[ "$(get 'c["model_name"]')" == hy4-flash ]] && ok "served as hy4-flash" || bad "model_name: $(get 'c.get("model_name")')"
[[ "$(get 'c["exe"]')" == /opt/maya/build/maya-hy4 ]] && ok "exe maya-hy4" || bad "exe: $(get 'c["exe"]')"
[[ "$(get 'c["installer"]["family"], c["installer"]["hy4_quant"], c["installer"]["quant"]')" == "('hy4', 'maya-l', 'Hy4-Flash-Maya-L')" ]] \
  && ok "installer record: family hy4, hy4_quant maya-l" || bad "installer: $(get 'c["installer"]')"
[[ "$(arg --max-context) $(has --mtp) $(has --kv) $(has --vision) $(has --profile) $(get '"env" in c')" == "32768 True False False False False" ]] \
  && ok "--max-context 32768, --mtp; no --kv / --vision / --profile, no env" || bad "args: $(get 'a') env: $(get 'c.get("env")')"
want="$(env $(clean) python3 -c "import sys; sys.path.insert(0, '$L'); import hy4_settings; print(hy4_settings.threads())")"
[[ "$(arg --cpu-threads)" == "$want" ]] && ok "--cpu-threads $want (threads())" || bad "--cpu-threads $(arg --cpu-threads), want $want"

echo "== 2. each setting lands on Maya's key"
write REASONING_EFFORT=none >/dev/null; [[ "$(get 'c["reasoning_effort"]')" == none ]] && ok "REASONING_EFFORT=none: thinking off" || bad "none: $(get 'c["reasoning_effort"]')"
out="$(write REASONING_EFFORT=low 2>&1)"; [[ "$(get 'c["reasoning_effort"]')" == high && "$out" == *"runs at high"* ]] \
  && ok "REASONING_EFFORT=low runs at high, and the boot says so" || bad "low: $(get 'c["reasoning_effort"]') / $out"
write REASONING_EFFORT=max ENABLE_THINKING=false >/dev/null; [[ "$(get 'c["reasoning_effort"]')" == none ]] && ok "ENABLE_THINKING=false wins" || bad "enable_thinking: $(get 'c["reasoning_effort"]')"
write THINKING_BUDGET=256 >/dev/null; [[ "$(get 'c["thinking_budget"]')" == 256 ]] && ok "THINKING_BUDGET -> thinking_budget" || bad "budget: $(get 'c.get("thinking_budget")')"
write TEMP=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0.05 PRESENCE_PENALTY=0.5 REPEAT_PENALTY=1.1 >/dev/null
[[ "$(get 'c["sampling"]')" == "{'temperature': 0.6, 'top_p': 0.95, 'min_p': 0.05, 'presence_penalty': 0.5, 'repetition_penalty': 1.1, 'top_k': 20}" ]] \
  && ok "the six sampler settings land in sampling" || bad "sampling: $(get 'c["sampling"]')"
write SERVED_NAME=llm LOOP_GUARD=false >/dev/null; [[ "$(get 'c["model_name"], c["loop_guard"]')" == "('llm', False)" ]] && ok "SERVED_NAME, LOOP_GUARD" || bad "name/guard: $(get 'c.get("model_name"), c.get("loop_guard")')"
write KV=q8 DRAFTING=off VISION=off THREADS=7 >/dev/null
[[ "$(arg --kv) $(has --mtp) $(arg --vision) $(arg --cpu-threads)" == "q8 False off 7" ]] && ok "KV=q8 -> --kv q8, DRAFTING=off drops --mtp, VISION=off -> --vision off, THREADS=7" || bad "args: $(get 'a')"
write HY4_EG_POPULATE=1 >/dev/null; [[ "$(get 'c.get("env")')" == "{'HY4_EG_POPULATE': '1'}" ]] && ok "HY4_EG_POPULATE goes into the config's env" || bad "env: $(get 'c.get("env")')"
write SPEC_N=4 MAX_TOKENS=100 >/dev/null; [[ "$(get 'sorted(k for k in c if k in ("spec_n", "max_tokens"))') $(has --draft)" == "[] False" ]] \
  && ok "SPEC_N / MAX_TOKENS write nothing" || bad "spec/max: $(get 'c')"

echo "== 3. the config is rewritten from scratch"
write THINKING_BUDGET=256 >/dev/null; write >/dev/null
[[ "$(get '"thinking_budget" in c')" == False ]] && ok "a setting removed is gone at the next boot" || bad "thinking_budget kept"

echo "== 4. bad values stop the boot; missing weights name the files"
for kv in REASONING_EFFORT=auto REASONING_EFFORT=HIGH ENABLE_THINKING=off TEMP=-1 TOP_P=0 TOP_K=0 KV=q4 KV=fp8 DRAFTING=yes VISION=1 THREADS=0 HY4_EG_POPULATE=2 CONTEXT=4095 SERVED_NAME="a b" LOOP_GUARD=yes; do
  env $(clean) "$kv" python3 "$L/hy4_settings.py" validate 2>/dev/null; r=$?
  [[ $r == 2 ]] && ok "$kv refused (exit 2)" || bad "$kv: exit $r"
done
out="$(env $(clean) REASONING_EFFORT=auto python3 "$L/hy4_prepare.py" --tag hy4-maya-l --src "$work/none" --data "$work/d" 2>&1)"; r=$?
[[ $r != 0 && "$out" == *REASONING_EFFORT* && ! -d "$work/d/hy4-maya-l/pack" ]] && ok "a bad setting stops main() before the files" || bad "main() with a bad setting: rc=$r $out"
out="$(env $(clean) python3 "$L/hy4_prepare.py" --tag hy4-maya-l --src "$work/none" --data "$work/d" 2>&1)"; r=$?
[[ $r != 0 && "$out" == *"Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf"* && "$out" == *"setup.sh"* ]] && ok "missing weights: the files and the download named" || bad "missing weights: rc=$r $out"

echo "== 5. threads()"
python3 -c "import sys; sys.path.insert(0, '$L'); import hy4_settings as s; assert s.threads(list(range(16))) == 8 and s.threads([0]) == 1 and s.threads([0, 1, 2]) == 1" \
  && ok "half the CPUs, at least 1" || bad "threads()"

echo "== 6. pack(): the shards by name, the vision tower and the MTP block, --out on the data volume"
out="$(python3 - "$L" "$work" <<'PY' 2>&1
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hy4_prepare as p
w = Path(sys.argv[2]) / "pk"; (w / "gguf").mkdir(parents=True)
for n in p.want("hy4-maya-l"):
    (w / "gguf" / n).write_bytes(b"x")
f = p.files_of("hy4-maya-l", w / "gguf")
seen = []
def fake_run(cmd, cwd=None):
    seen.append(cmd)
    out = Path(cmd[cmd.index("--out") + 1]); (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for n in ("hy4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
        (out / n).write_text("x")
    return types.SimpleNamespace(returncode=0)
p.subprocess.run = fake_run
d = w / "data"; d.mkdir()
pk = p.pack(f, d, False)
cmd = seen[-1]
print("files", cmd[cmd.index("--files") + 1:] == ["Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf", "Hy4-Flash-Maya-L-IQ4_XS-00002-of-00002.gguf"])
print("mmproj", cmd[cmd.index("--mmproj") + 1] == "mmproj-Hy4-Flash-Maya-XS.gguf")
print("mtp", cmd[cmd.index("--mtp") + 1] == "Hy4-Flash-mtp-BF16.gguf")
print("out", Path(cmd[cmd.index("--out") + 1]).parent == d and pk == d / "pack")
p.pack(f, d, False)
print("rebuilt-when-unchanged", len(seen) > 1)
PY
)"
[[ "$out" == *"files True"* && "$out" == *"mmproj True"* && "$out" == *"mtp True"* && "$out" == *"out True"* ]] \
  && ok "pack_gguf.py gets both shards, the vision tower, the MTP block and --out on /data" || bad "pack command: $out"
[[ "$out" == *"rebuilt-when-unchanged False"* ]] && ok "an up-to-date pack isn't built again" || bad "pack rebuilt: $out"

echo "== 7. verify(): the SHA-256 table stops a damaged file; a match is recorded"
out="$(python3 - "$L" "$work" <<'PY' 2>&1
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hy4_prepare as p
w = Path(sys.argv[2]) / "v"; w.mkdir()
f = w / "a.gguf"; f.write_bytes(b"good")
p.want = lambda tag: {"a.gguf": hashlib.sha256(b"good").hexdigest()}
p.verify("t", [f], w)                               # matches: recorded
calls = []
p.sha256 = lambda x: calls.append(x) or "never"     # a second verify must not hash again
p.verify("t", [f], w)
print("rehashed" if calls else "not rehashed")
f.write_bytes(b"bad!")
st = f.stat()
os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
p.sha256 = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
p.verify("t", [f], w)
print("NOT STOPPED")
PY
)"
command grep -q "every file matches" <<<"$out" && command grep -q "not rehashed" <<<"$out" \
  && ok "a matching file is recorded and not hashed again" || bad "verify record: $out"
command grep -q "SHA-256 mismatch" <<<"$out" && ! command grep -q "NOT STOPPED" <<<"$out" \
  && ok "a changed file is hashed again and stops the boot" || bad "verify mismatch: $out"

echo "== 8. the pinned engine reads every key a setting writes (engine-config-reads.json; no docker)"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/lib/profiles/engine_config_reads.py check --engine strata-maya:hy4 || rc=1

exit $rc
