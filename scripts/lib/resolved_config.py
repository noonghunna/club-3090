#!/usr/bin/env python3
"""resolved_config.py — what the ENGINE ACTUALLY RAN, and how it differs from
the shipped recipe (club-3090#1265).

WHY THIS EXISTS
---------------
`report.sh` profiled the rig thoroughly and captured NOTHING about the engine's
configuration: eight `docker inspect` calls, never for `Config.Cmd`,
`Config.Entrypoint` or `Config.Env`.  So an unexpected number in a report was
unattributable — we could not tell a shipped recipe from a modified one.  #1261
is the worked example: a clean `--full` report, and the three findings that
mattered (`--mem-fraction-static 0.82` vs the recipe's 0.91,
`--chunked-prefill-size 2048` vs 1024, and the drafter identity) came only from
a boot log the reporter happened to paste into a comment.

⛔ WHY WE DO NOT DUMP THE COMPOSE FILE
-------------------------------------
The obvious implementation is wrong, not merely incomplete:

1. A compose is a TEMPLATE, not a configuration.  Ours are almost entirely
   `${VAR:-default}`, so the text shows DEFAULTS.  A user who exports
   `MEM_FRACTION=0.82` changes not one byte of the file — the exact case this
   module exists to detect stays invisible.
2. The launchers OVERRIDE the compose.  `switch.sh`/`launch.sh` resolve the
   engine image from `scripts/lib/profiles/engines/<engine>.yml` `install.spec`
   and inject it, beating the compose's own `image:` default.  A compose dump
   would therefore print the WRONG engine image — institutionalising a trap this
   stack already documents.

MEASURED, not assumed (2026-09-12, this rig, a throwaway 2-line compose):

    compose text                  docker compose config       .Config.Cmd
    ${MEM_FRACTION:-0.91}   ->    host-sub=0.91          ->   host-sub=0.82
    $${MEM_FRACTION:-...}   ->    $${MEM_FRACTION:-...}  ->   ${MEM_FRACTION:-...}

i.e. single-`$` interpolation happens on the HOST at launch, so `.Config.Cmd`
carries the RESOLVED value; `docker compose config` run in a clean environment
re-renders the same script with the recipe's defaults; and `$$` survives into
`config` output but is unescaped to `$` in the real container.  That last row is
why `_normalise_script` exists — without it every `$$` line would read as a
difference.

WHAT WE CAPTURE, in increasing order of authority
-------------------------------------------------
(a) What the CONTAINER received — `docker inspect` `.Config.{Image,Entrypoint,
    Cmd,Env}`.  Authoritative for launcher injection and env overrides.
(b) What the ENGINE resolved — its own startup dump, where an `auto` gets
    concretised (`--kv-cache-dtype auto -> fp8_e4m3` is visible nowhere else).
    ELIDED to the keys that matter by default; `--engine-args` prints the dump.
(c) ⭐ A DIFF against the shipped recipe — the actual ask.  Resolve the slug
    through the registry, re-render the recipe in a clean environment, print
    only the delta.  Where the slug is unrecognised, SAY SO — never print
    nothing and let it read as "no differences".

SECRETS: ALLOWLIST, NEVER A DENYLIST
------------------------------------
`.Config.Env` carries `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, gateway keys.  A
denylist fails open the first time someone adds a new secret, so a name must
MATCH the allowlist to print at all.  Two notes on the shape of that allowlist:

  * It is NOT derived from the composes' own `environment:` lists, even though
    that would self-maintain: 75 shipped composes declare
    `HUGGING_FACE_HUB_TOKEN` there, so a compose-derived allowlist would leak
    the token by construction.
  * The prefix families (`VLLM_`, `LLAMA_ARG_`, `SGLANG_`, …) each contain a
    real credential var — llama-server reads `LLAMA_ARG_API_KEY`, vLLM reads
    `VLLM_API_KEY`.  `_NAME_VETO` removes those shapes from within an allowed
    family.  It is a SECOND GATE, not the mechanism: a name outside the
    allowlist never prints no matter what the veto says.

The allowlist is enforced regardless of `--no-redact`: that flag is documented
as disabling PATH/HOST/USER redaction, and no report has ever carried engine env
at all, so holding the line here regresses nothing.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Env allowlist
# --------------------------------------------------------------------------

# Exact names: the tuning knobs our shipped composes pass through, plus the
# handful of launcher/runtime vars that change what the engine does.  Grown by
# PR; `test-report-resolved-config.sh` asserts the filter's shape, not this
# list's contents, so adding a knob here is a one-line change.
_ALLOW_EXACT = frozenset(
    """
    ASYNC_SCHED ATTENTION_BACKEND BIND_HOST CHUNKED_PREFILL_SIZE CLUB3090_RESTART
    CONTEXT_LENGTH CUDA_DEVICE_MAX_CONNECTIONS CUDA_DEVICE_ORDER
    CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES DECODE_GRANULARITY DRAFTER_DIR
    DRAFT_QUANT ENABLE_THINKING ESTATE_CONTAINER ESTATE_GPUS ESTATE_PORT
    HF_HUB_OFFLINE INSTRUCT KV_CACHE_DTYPE KV_CACHE_MEMORY_BYTES MAMBA_BLOCK_SIZE
    MAMBA_CACHE_DTYPE MAMBA_SSM_DTYPE MAX_RUNNING_REQUESTS MEM_FRACTION MIN_P
    MMPROJ MODEL_DIR MTP_ACCEPT_MIN NGRAM_N_MATCH NGRAM_N_MIN NUM_SPEC_TOKENS
    NVIDIA_VISIBLE_DEVICES NVLINK_MODE OMP_NUM_THREADS PORT PRESENCE_PENALTY
    PYTHONPATH PYTHONUTF8 PYTORCH_CUDA_ALLOC_CONF QUANTIZATION REASONING
    REASONING_EFFORT REASONING_PARSER REPEAT_PENALTY SPEC SPEC_N SPEC_N_MAX TEMP
    TEMPERATURE THINKING THREADS TOOL_CALL_PARSER TOP_K TOP_P TRANSFORMERS_OFFLINE
    TRITON_CACHE_DIR W4A8 W4A8_PATCH_DIR
    """.split()
)

# Prefix families.  Each is a bulk engine-tuning namespace where new members
# appear constantly and enumerating them would rot; _NAME_VETO covers the
# credential-shaped members that live inside them.
_ALLOW_PREFIX = (
    "GENESIS_",
    "GGML_",
    "LLAMA_ARG_",
    "LLAMA_GRAPH_",
    "LMCACHE_",
    "NCCL_",
    "SGLANG_",
    "TORCHDYNAMO_",
    "TORCHINDUCTOR_",
    "TORCH_",
    "VLLM_",
)

# Second gate (NOT the mechanism — see the module docstring).  Applied to names
# that already passed the allowlist.
_NAME_VETO = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|_KEY$|^KEY_|AUTH|SESSION"
    r"|COOKIE|PRIVATE|SALT|SIGNATURE)",
    re.I,
)

# Value-shape scrub, applied last.  Keeps the NAME (so the reader knows the var
# was set) and replaces the value, rather than dropping the row silently.
_VALUE_VETO = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

_MAX_VALUE_CHARS = 200
_MAX_ENV_ROWS = 60


def env_allowed(name):
    """True when NAME may appear in a report at all."""
    if not name:
        return False
    if _NAME_VETO.search(name):
        return False
    if name in _ALLOW_EXACT:
        return True
    return any(name.startswith(p) for p in _ALLOW_PREFIX)


def scrub_value(value):
    """Token-shaped value -> placeholder, then truncate.

    Applied everywhere a value is printed — the env listing AND the env-diff
    table. They used to differ: the diff table printed raw, so an allowlisted
    var holding a token-shaped value was scrubbed in one block and verbatim in
    the other. Belt and braces only works if every strap is fastened.
    """
    for rx in _VALUE_VETO:
        if rx.search(value):
            value = rx.sub("<REDACTED-SHAPE>", value)
    if len(value) > _MAX_VALUE_CHARS:
        value = value[:_MAX_VALUE_CHARS] + "…(truncated)"
    return value


def filter_env(pairs):
    """[(name, value)] -> (kept_rows, n_withheld, n_declared_empty).

    Values are scrubbed then truncated.  Allowlisted vars with an EMPTY value
    are counted, not listed: our composes declare a dozen bare `- VAR`
    passthroughs per service, and an unset one carries no information beyond
    "the reporter did not override it" — listing them would be a dozen lines of
    noise in a section whose whole selling point is that it is bounded.
    """
    kept, withheld, empty = [], 0, 0
    for name, value in pairs:
        if not env_allowed(name):
            withheld += 1
            continue
        if value == "":
            empty += 1
            continue
        kept.append((name, scrub_value(value)))
    kept.sort()
    if len(kept) > _MAX_ENV_ROWS:
        withheld += len(kept) - _MAX_ENV_ROWS
        kept = kept[:_MAX_ENV_ROWS]
    return kept, withheld, empty


def _split_env_line(line):
    name, sep, value = line.partition("=")
    return (name, value) if sep else (line, "")


# --------------------------------------------------------------------------
# Command / flag extraction
# --------------------------------------------------------------------------

_COMMENT = re.compile(r"^\s*#")
_FLAG_RX = re.compile(
    r"(?<![\w\-])--([A-Za-z][A-Za-z0-9._-]*)"
    r"(?:[=\s]+(\"[^\"\n]*\"|'[^'\n]*'|[^\s\\]+))?"
)


def _normalise_script(text):
    """Recipe-side normalisation.

    `docker compose config` re-emits a compose file, so the `$$` escape survives
    in its output; the daemon receives the unescaped `$`.  Without this every
    `$$` line reads as a difference.  MEASURED, see the module docstring.
    """
    return text.replace("$$", "$")


def looks_like_script(argv):
    """A bash -c payload arrives as a 1-element argv carrying a whole script."""
    if len(argv) != 1:
        return False
    only = argv[0]
    return "\n" in only or only.lstrip().startswith(("set -", "#!"))


def summarise_argv(argv):
    """One-line, paste-safe description of an entrypoint/command argv.

    A bash -c payload is a whole script (11 KB on the SGLang slugs); printing
    two of them side by side to say "these differ" costs more than the whole
    rest of the section. Identity by sha256 + shape is enough — the flag diff
    above it is the readable form.
    """
    if not argv:
        return "(unset)"
    # Script-shaped either as a bare 1-element payload (SGLang / llama.cpp
    # composes) or as `/bin/bash -c <script>` (the vLLM composes, whose
    # entrypoint carries the W4A8 install shim). Both must summarise, or the
    # "Entrypoint:" bullet spills a multi-line script into a bullet list.
    idx = next((i for i, a in enumerate(argv) if "\n" in a), None)
    if idx is None and looks_like_script(argv):
        idx = 0
    if idx is not None:
        b = argv[idx].encode("utf-8")
        prefix = (" ".join(argv[:idx]) + " ") if idx else ""
        return "`%sinline script` — %d lines, %d bytes, sha256 `%s`" % (
            prefix, len(argv[idx].splitlines()), len(b), hashlib.sha256(b).hexdigest()[:16])
    joined = " ".join(argv)
    if len(joined) > 200:
        joined = joined[:200] + "…"
    return "`%s`" % joined


def argv_lines(argv):
    """Flat argv regrouped one FLAG per line — faithful (no token is dropped or
    rewritten) but readable in a pasted issue body, unlike one token per line."""
    out, cur = [], []
    for tok in argv:
        if tok.startswith("--") and cur:
            out.append(" ".join(cur))
            cur = [tok]
        else:
            cur.append(tok)
    if cur:
        out.append(" ".join(cur))
    return out


def _strip_quotes(v):
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def flags_from_argv(argv):
    """Flat argv (the vLLM compose shape): --flag plus ALL following non-flag
    tokens.

    Multi-value flags are real here — `--served-model-name qwen3.6-27b
    qwen3.6-27b-autoround` takes two — so taking only the next token would make
    a change to the second value invisible.  Leading positionals (the `vllm
    serve <model>` shape) are captured under a pseudo-name so a swapped model
    path is not missed either.  Together these make argv coverage COMPLETE,
    which is what lets the caller skip the line-residual check for argv.
    """
    out = []
    i = 0
    lead = []
    while i < len(argv) and not argv[i].startswith("--"):
        lead.append(argv[i])
        i += 1
    if lead:
        out.append(("<positional>", " ".join(lead)))
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        name, eq, inline = tok[2:].partition("=")
        if eq:
            out.append((name, inline))
            i += 1
            continue
        vals = []
        i += 1
        while i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        out.append((name, " ".join(vals)))
    return out


def flags_from_script(text):
    """Shell-script payload (the SGLang / llama.cpp compose shape).

    Comment lines are dropped first — several composes name flags inside `# ⚠️`
    prose ("--speculative-draft-model-quantization IS LOAD-BEARING"), which
    would otherwise extract as a flag whose value is the next English word.
    Anything spurious that survives appears IDENTICALLY on both sides of the
    diff and therefore cancels; the diff, not the extraction, is the contract.
    """
    body = "\n".join(ln for ln in text.splitlines() if not _COMMENT.match(ln))
    return [(m.group(1), _strip_quotes(m.group(2) or "")) for m in _FLAG_RX.finditer(body)]


def flag_map(argv):
    """argv (from .Config.Cmd or a rendered recipe) -> {flag: [values]}."""
    pairs = flags_from_script(argv[0]) if looks_like_script(argv) else flags_from_argv(argv)
    out = {}
    for name, value in pairs:
        out.setdefault(name, []).append(value)
    return out


def body_lines(argv):
    """Substantive (non-comment, non-blank) launch-script lines, whitespace
    normalised — the residual check that catches edits a flag diff cannot see
    (a changed model path, a deleted guard).

    Empty for a flat argv: `flags_from_argv` already covers every token there,
    so running a residual check over it would double-report each changed value.
    """
    if not looks_like_script(argv):
        return []
    out = []
    for ln in argv[0].splitlines():
        if _COMMENT.match(ln):
            continue
        s = " ".join(ln.split())
        if s:
            out.append(s)
    return out


# --------------------------------------------------------------------------
# docker / registry plumbing
# --------------------------------------------------------------------------


def _run(cmd, env=None, timeout=60):
    """(rc, stdout, stderr). Never raises; a missing binary is rc=127."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "not found: %s" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "", "timed out: %s" % " ".join(cmd)
    except OSError as exc:  # noqa: BLE001 - surfaced, never swallowed
        return 126, "", str(exc)


def docker_inspect(container):
    rc, out, err = _run(["docker", "inspect", container, "--format", "{{json .}}"])
    if rc != 0 or not out.strip():
        return None, (err.strip() or "docker inspect exited %d" % rc)
    try:
        return json.loads(out), None
    except ValueError as exc:
        return None, "docker inspect returned unparseable JSON: %s" % exc


def load_registry(root):
    """COMPOSE_REGISTRY via its stdlib-only loader.

    Deliberately NOT registry-emit.sh --json: that path may require PyYAML, and
    a community rig without it is exactly the rig whose report we most need.
    """
    sys.path.insert(0, str(Path(root) / "scripts" / "lib" / "profiles"))
    try:
        from compose_registry import COMPOSE_REGISTRY  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        return None, "compose registry unreadable: %s" % exc
    return COMPOSE_REGISTRY, None


def engine_profile_image(root, engine_id):
    """`install.spec` for an engine id — the image the LAUNCHERS inject.

    Plain-text parse, no PyYAML (same constraint as the launcher table path).
    The file basename is not the id for every profile (llama-cpp-mainline.yml
    declares `id: llama-cpp-local`), so match on the declared `id:`.
    """
    d = Path(root) / "scripts" / "lib" / "profiles" / "engines"
    if not d.is_dir():
        return None
    for path in sorted(d.glob("*.yml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.search(r"^id:\s*(\S+)\s*$", text, re.M)
        if not m or m.group(1) != engine_id:
            continue
        s = re.search(r"^install:\s*$.*?^\s+spec:\s*(\S+)\s*$", text, re.M | re.S)
        return s.group(1) if s else None
    return None


def resolve_slug(root, container, info):
    """(slug, compose_path, how, note) — any may be None.

    Evidence order: the compose project label (exact, then suffix for a
    different checkout root), then the compose's own `container_name` default.
    `how` is printed so a reader can judge the match rather than trust it.
    """
    registry, err = load_registry(root)
    if registry is None:
        return None, None, None, err

    labels = (info.get("Config") or {}).get("Labels") or {}
    raw = labels.get("com.docker.compose.project.config_files") or ""
    files = [f for f in raw.split(",") if f.strip()]

    root_p = Path(root).resolve()
    for f in files:
        fp = Path(f.strip())
        for slug, entry in registry.items():
            cp = entry.get("compose_path")
            if not cp:
                continue
            try:
                same = (root_p / cp).resolve() == fp.resolve()
            except OSError:
                same = False
            if same:
                return slug, cp, "compose project label (exact path)", None
    for f in files:
        norm = str(Path(f.strip())).replace(os.sep, "/")
        for slug, entry in registry.items():
            cp = entry.get("compose_path")
            if cp and norm.endswith("/" + cp):
                return slug, cp, "compose project label (path suffix, other checkout)", None

    if container:
        rx = re.compile(r"^\s*container_name:\s*(?:\$\{[A-Z_0-9]+:-)?([^\s}\"']+)", re.M)
        for slug, entry in sorted(registry.items()):
            cp = entry.get("compose_path")
            if not cp:
                continue
            try:
                text = (root_p / cp).read_text(encoding="utf-8")
            except OSError:
                continue
            m = rx.search(text)
            if m and m.group(1) == container:
                return slug, cp, "compose container_name default", None

    if files:
        return None, None, None, "compose file is not a shipped recipe: %s" % ", ".join(files)
    return None, None, None, "container carries no docker-compose project labels (started by `docker run`, or by hand)"


def render_recipe(root, compose_path, container):
    """The shipped recipe re-rendered in a CLEAN environment.

    `env -i` + `--env-file /dev/null` is the whole point: every `${VAR:-default}`
    falls to its DEFAULT, so what comes back is the recipe as shipped, not the
    recipe as this rig happens to be configured.
    """
    full = Path(root) / compose_path
    if not full.is_file():
        return None, "recipe not found on disk: %s" % compose_path
    clean = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp")}
    rc, out, err = _run(
        ["docker", "compose", "--env-file", "/dev/null", "-f", str(full), "config", "--format", "json"],
        env=clean,
    )
    if rc != 0 or not out.strip():
        return None, "could not render the recipe (`docker compose config` exited %d): %s" % (
            rc,
            " ".join((err or "").split())[:200],
        )
    try:
        doc = json.loads(out)
    except ValueError as exc:
        return None, "recipe render returned unparseable JSON: %s" % exc
    services = doc.get("services") or {}
    if not services:
        return None, "recipe render contains no services"
    chosen = None
    for name, svc in services.items():
        if container and svc.get("container_name") == container:
            chosen = (name, svc)
            break
    if chosen is None:
        name = sorted(services)[0]
        chosen = (name, services[name])
    return chosen, None


def env_diff(recipe_env, actual_pairs):
    """(rows, n_withheld) — declared env vars whose value differs from the recipe.

    WHY THIS EXISTS SEPARATELY FROM THE FLAG DIFF: not every knob reaches argv.
    The llama.cpp composes declare `SPEC` / `SPEC_N` and consume them INSIDE the
    container (`$$SPEC`), so an override changes the engine's behaviour while
    `.Config.Cmd` stays byte-identical — a "stock recipe" verdict taken from
    flags alone would be a false clean.

    Scope is the recipe's OWN `environment:` names: anything else in
    `.Config.Env` is baked into the image, not a user decision.  Names that are
    not on the allowlist are COUNTED, never named and never compared — the
    allowlist is what defines "tuning knob", and a differing
    `HUGGING_FACE_HUB_TOKEN` is a credential, not a configuration change.
    """
    actual = dict(actual_pairs)
    rows, withheld = [], 0
    for name in sorted(recipe_env or {}):
        if not env_allowed(name):
            withheld += 1
            continue
        want = recipe_env.get(name)
        # compose renders a bare, unset passthrough as null; the daemon gives it
        # to the container as "". Same state, two spellings.
        want = "" if want is None else str(want)
        got = actual.get(name, "")
        if got != want:
            rows.append((name, scrub_value(got), scrub_value(want)))
    return rows, withheld


def _as_argv(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return shlex.split(str(value)) if "\n" not in str(value) else [str(value)]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_MAX_DIFF_ROWS = 40
_MAX_RESIDUAL = 5
_MAX_ARGV_TOKENS = 60
_MAX_DUMP_CHARS = 20000

# (b) The engine's OWN startup dump — where an `auto` becomes a concrete value.
# Elided by default to these lines; --engine-args prints the matched dump whole.
_DUMP_PATTERNS = {
    "sglang": (r"server_args=", r"^\s*Init torch distributed", r"KV Cache is allocated",
               r"max_total_num_tokens="),
    "vllm": (r"non-default args:", r"Initializing a V1 LLM engine", r"Using .* KV cache",
             r"GPU KV cache size:", r"Maximum concurrency for"),
    "llamacpp": (r"^system_info:", r"^build:", r"^main: ", r"llama_context:",
                 r"llama_kv_cache", r"^srv .*params"),
}
# The keys worth surfacing from a one-line `server_args=ServerArgs(...)` dump
# (~8 KB on one line — see the size argument in #1265).
_SGLANG_KEYS = (
    "model_path", "quantization", "tp_size", "context_length", "mem_fraction_static",
    "chunked_prefill_size", "kv_cache_dtype", "attention_backend", "mamba_ssm_dtype",
    "max_running_requests", "speculative_algorithm", "speculative_draft_model_path",
    "speculative_num_draft_tokens", "speculative_dflash_block_size",
    "speculative_num_steps", "speculative_eagle_topk", "reasoning_parser",
    "tool_call_parser", "served_model_name",
)


# The startup dump is at the HEAD of the boot log, but a long-lived engine's log
# can be hundreds of MB — and `docker logs` has no `--head`. So read a bounded
# PREFIX and stop, rather than slurping the whole thing into memory.
_LOG_PREFIX_BYTES = 8 * 1024 * 1024


def _boot_log_prefix(container):
    """(text, truncated, note) — at most _LOG_PREFIX_BYTES from the log's head."""
    try:
        proc = subprocess.Popen(
            ["docker", "logs", container],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return "", False, "docker not found"
    except OSError as exc:
        return "", False, str(exc)
    chunks, total = [], 0
    try:
        while total < _LOG_PREFIX_BYTES:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return b"".join(chunks).decode("utf-8", "replace"), total >= _LOG_PREFIX_BYTES, None


def engine_dump(container, kind, full):
    """(lines, note). `note` is set on every path that yields nothing, so an
    empty capture never reads the same as a clean one."""
    pats = _DUMP_PATTERNS.get(kind)
    if not pats:
        return [], "engine kind `%s` has no known startup dump to read" % (kind or "unknown")
    blob, truncated, lerr = _boot_log_prefix(container)
    if lerr:
        return [], "could not read the boot log (%s)" % lerr
    if not blob.strip():
        return [], "boot log is empty (log driver not `json-file`, or the log was rotated)"
    rxs = [re.compile(p) for p in pats]
    hits = []
    for ln in blob.splitlines():
        if any(r.search(ln) for r in rxs):
            hits.append(ln.rstrip())
    if not hits:
        return [], ("no engine startup dump matched in the first %d MB of the boot log"
                    % (_LOG_PREFIX_BYTES // (1024 * 1024)) if truncated else
                    "no engine startup dump matched in the boot log (rotated past it, "
                    "or a newer engine format)")
    if full:
        joined = "\n".join(hits)
        if len(joined) > _MAX_DUMP_CHARS:
            joined = joined[:_MAX_DUMP_CHARS] + "\n…(truncated at %d chars)" % _MAX_DUMP_CHARS
        return joined.splitlines(), None
    out_lines = []
    for ln in hits:
        if kind == "sglang" and "server_args=" in ln:
            found = []
            for key in _SGLANG_KEYS:
                m = re.search(r"\b%s=('[^']*'|\"[^\"]*\"|[^,)\s]+)" % re.escape(key), ln)
                if m:
                    found.append("%s=%s" % (key, m.group(1)))
            out_lines.append("server_args (elided to %d keys; --engine-args for the full dump):" % len(found))
            out_lines.extend("  " + f for f in found)
        else:
            out_lines.append(ln if len(ln) <= 300 else ln[:300] + "…")
    return out_lines[:40], None


_MAX_FLAG_ROWS = 60


def _render_flag_list(add, running):
    """Print the extracted launch flags outright.

    Used where there is NO recipe to diff against — a hand-rolled compose
    (#1261) or a recipe we could not render.  Without this the section would
    promise the reporter's flags and then show nothing, which is precisely the
    gap #1265 exists to close: an unrecognised launch is the case where argv is
    the ONLY evidence we have.
    """
    names = sorted(running)
    if not names:
        add("- _No `--flag` tokens found in the container's command._")
        return
    add("- **Launch flags extracted from `.Config.Cmd`** (%d):" % len(names))
    add("")
    add("```")
    for name in names[:_MAX_FLAG_ROWS]:
        for value in running[name]:
            v = value if len(value) <= 120 else value[:120] + "…"
            add("--%s%s" % (name, (" " + v) if v else ""))
    if len(names) > _MAX_FLAG_ROWS:
        add("… (%d more flags)" % (len(names) - _MAX_FLAG_ROWS))
    add("```")


def render(root, container, kind, full_engine_args):
    L = []
    add = L.append

    if not container:
        add("_No engine container resolved — nothing to introspect. "
            "Start one with `bash scripts/launch.sh` and re-run._")
        return "\n".join(L)
    if container == "none":
        add("_Host engine build (`CONTAINER=none`) — there is no container, so the "
            "resolved config and the recipe diff cannot be captured._")
        return "\n".join(L)

    info, err = docker_inspect(container)
    if info is None:
        add("_Could not inspect `%s` — %s. No resolved config captured._" % (container, err))
        return "\n".join(L)

    cfg = info.get("Config") or {}
    image = cfg.get("Image") or "(unset)"
    entrypoint = _as_argv(cfg.get("Entrypoint"))
    cmd = _as_argv(cfg.get("Cmd"))
    env_pairs = [_split_env_line(e) for e in (cfg.get("Env") or [])]
    state = (info.get("State") or {}).get("Status") or "unknown"

    slug, compose_path, how, note = resolve_slug(root, container, info)
    running = flag_map(cmd)

    # ---- (c) the recipe diff — the headline, so it goes first ----
    add("### Shipped-recipe comparison")
    add("")
    if slug:
        add("- **Slug:** `%s` — recipe `%s`" % (slug, compose_path))
        add("- **Matched by:** %s" % how)
        rcgit, gout, _ = _run(["git", "-C", str(root), "status", "--porcelain", "--", compose_path])
        if rcgit == 0 and gout.strip():
            add("- ⚠️ **The local copy of this recipe has uncommitted modifications** — "
                "the diff below is against the WORKING-TREE recipe, not the shipped one.")
        chosen, rerr = render_recipe(root, compose_path, container)
        if chosen is None:
            add("- ⚠️ **Recipe diff unavailable:** %s" % rerr)
            add("- Nothing to diff against, so the flags are reported raw instead:")
            _render_flag_list(add, running)
        else:
            _svc_name, svc = chosen
            r_image = svc.get("image") or "(unset)"
            # ⚠️ The entrypoint needs the SAME `$$` normalisation as the command:
            # the vLLM composes end theirs with `exec vllm serve "$$@"`, so
            # comparing the raw render against the container reported a phantom
            # entrypoint difference on every stock vLLM slug.
            r_entry = [_normalise_script(c) for c in _as_argv(svc.get("entrypoint"))]
            r_cmd = [_normalise_script(c) for c in _as_argv(svc.get("command"))]

            recipe = flag_map(r_cmd)
            rows = []
            for name in sorted(set(running) | set(recipe)):
                a, b = running.get(name), recipe.get(name)
                if a == b:
                    continue
                rows.append((name,
                             ", ".join("`%s`" % v if v else "_(bare)_" for v in a) if a else "_absent_",
                             ", ".join("`%s`" % v if v else "_(bare)_" for v in b) if b else "_absent_"))

            # Residual check: script payloads only (body_lines is empty for argv,
            # which flags_from_argv already covers token-for-token).
            r_body, a_body = body_lines(r_cmd), body_lines(cmd)
            residual = [ln for ln in a_body if ln not in r_body]
            # Lines that differ only in a flag value are already in `rows`.
            residual = [ln for ln in residual if not any("--" + n in ln for n, _, _ in rows)]

            erows, ewithheld = env_diff(svc.get("environment") or {}, env_pairs)
            # Most knobs are interpolated by compose ON THE HOST, so an override
            # already shows as a flag difference; listing it twice is noise. Keep
            # only the env rows whose value surfaces NOWHERE in the flag diff —
            # which is exactly the false-clean class this table exists to catch.
            _flag_vals = {v for _n, _a, _b in rows for v in running.get(_n, [])}
            erows = [e for e in erows if e[1] not in _flag_vals or not e[1]]

            if not rows and not residual and not erows and entrypoint == r_entry:
                add("- ✅ **Stock recipe** — launch flags, entrypoint, launch script and every "
                    "allowlisted env knob are identical to the shipped recipe rendered with its "
                    "own defaults.")
            else:
                if rows:
                    add("- ⚠️ **%d flag(s) differ from the shipped recipe:**" % len(rows))
                    add("")
                    add("| flag | running | recipe |")
                    add("|---|---|---|")
                    for name, a, b in rows[:_MAX_DIFF_ROWS]:
                        add("| `--%s` | %s | %s |" % (name, a, b))
                    if len(rows) > _MAX_DIFF_ROWS:
                        add("| … | _(%d more)_ | |" % (len(rows) - _MAX_DIFF_ROWS))
                    add("")
                if entrypoint != r_entry:
                    add("- ⚠️ **Entrypoint differs:** running %s vs recipe %s"
                        % (summarise_argv(entrypoint), summarise_argv(r_entry)))
                if erows:
                    add("- ⚠️ **%d env knob(s) differ from the shipped recipe and do not appear "
                        "in the flags above** — they are read inside the container, so argv "
                        "alone would have reported this launch as stock:" % len(erows))
                    add("")
                    add("| env var | running | recipe |")
                    add("|---|---|---|")
                    for name, got, want in erows[:_MAX_DIFF_ROWS]:
                        add("| `%s` | %s | %s |"
                            % (name,
                               "`%s`" % got if got else "_(unset)_",
                               "`%s`" % want if want else "_(unset)_"))
                    if len(erows) > _MAX_DIFF_ROWS:
                        add("| … | _(%d more)_ | |" % (len(erows) - _MAX_DIFF_ROWS))
                    add("")
                if residual:
                    add("- ⚠️ **%d further launch-script line(s) differ** "
                        "(flag changes already listed above are excluded):" % len(residual))
                    add("")
                    add("```")
                    for ln in residual[:_MAX_RESIDUAL]:
                        add(ln if len(ln) <= 200 else ln[:200] + "…")
                    if len(residual) > _MAX_RESIDUAL:
                        add("… (%d more)" % (len(residual) - _MAX_RESIDUAL))
                    add("```")

            if ewithheld:
                add("- _(%d declared env var(s) are off the allowlist — credentials and the like. "
                    "Not compared, not named, and not counted against the verdict.)_" % ewithheld)

            entry = (load_registry(root)[0] or {}).get(slug) or {}
            eng_id = entry.get("engine")
            prof_image = engine_profile_image(root, eng_id) if eng_id else None
            if prof_image:
                verdict = "matches" if image == prof_image else "⚠️ **differs from**"
                add("- **Engine image:** `%s` — %s the engine profile `%s` (`%s`)"
                    % (image, verdict, eng_id, prof_image))
                if r_image not in (prof_image, "(unset)") and image == prof_image:
                    add("  - (the compose's own `image:` default is `%s`; the launchers inject "
                        "the profile's, which is why a compose dump would report the wrong image)"
                        % r_image)
            else:
                verdict = "matches" if image == r_image else "⚠️ **differs from**"
                add("- **Engine image:** `%s` — %s the recipe's `image:` default (`%s`). "
                    "Engine `%s` pins no `install.spec`, so the compose default is the pin truth."
                    % (image, verdict, r_image, eng_id or "unknown"))
    else:
        add("- ⚠️ **Unrecognised — not a shipped recipe.** %s" % (note or "no evidence available"))
        add("- There is nothing to diff against, so the flags are reported raw. Every value "
            "here is the reporter's own configuration, not a club-3090 default.")
        _render_flag_list(add, running)
    add("")

    # ---- (a) what the container received ----
    add("### Container config (`docker inspect`)")
    add("")
    add("- **Container:** `%s` (state: %s, engine kind: `%s`)" % (container, state, kind or "unknown"))
    add("- **Image:** `%s`" % image)
    add("- **Entrypoint:** %s" % summarise_argv(entrypoint))
    if looks_like_script(cmd):
        add("- **Cmd:** %s — not printed; the recipe diff above is the readable form"
            % summarise_argv(cmd))
    elif cmd:
        lines = argv_lines(cmd)
        add("- **Cmd:**")
        add("")
        add("```")
        for ln in lines[:_MAX_ARGV_TOKENS]:
            add(ln if len(ln) <= 200 else ln[:200] + "…")
        if len(lines) > _MAX_ARGV_TOKENS:
            add("… (%d more)" % (len(lines) - _MAX_ARGV_TOKENS))
        add("```")
    else:
        add("- **Cmd:** (unset — the image's own default runs)")

    kept, withheld, empty = filter_env(env_pairs)
    add("- **Env (allowlist — %d shown, %d set-but-empty, %d withheld):** an env var must "
        "MATCH the allowlist to print; this holds under `--no-redact` too."
        % (len(kept), empty, withheld))
    if kept:
        add("")
        add("```")
        for name, value in kept:
            add("%s=%s" % (name, value))
        add("```")
    else:
        add("  - _none of the container's %d env vars carries an allowlisted, non-empty value._"
            % len(env_pairs))
    add("")

    # ---- (b) what the engine resolved ----
    add("### Engine-resolved values (engine's own startup dump)")
    add("")
    lines, dnote = engine_dump(container, kind, full_engine_args)
    if dnote:
        add("_%s._" % dnote)
    else:
        add("```")
        L.extend(lines)
        add("```")
        if not full_engine_args:
            add("")
            add("_Elided. Re-run with `--engine-args` for the engine's full startup dump._")
    return "\n".join(L)


def main(argv):
    # This module's output is unicode (⚠ ✅ —) and report.sh pipes it into
    # `redact`. Under a real single-byte locale the ENCODE side raises, so pin
    # stdout explicitly — the backstop the repo mandates behind PYTHONUTF8 (#777).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if argv[:1] == ["env-filter"]:
        pairs = [_split_env_line(ln.rstrip("\n")) for ln in sys.stdin if ln.strip()]
        kept, withheld, empty = filter_env(pairs)
        for name, value in kept:
            print("%s=%s" % (name, value))
        print("# set-but-empty: %d" % empty)
        print("# withheld: %d" % withheld)
        return 0
    if argv[:1] == ["section"]:
        root = container = kind = ""
        full = False
        rest = argv[1:]
        i = 0
        while i < len(rest):
            if rest[i] == "--root":
                root = rest[i + 1]; i += 2
            elif rest[i] == "--container":
                container = rest[i + 1]; i += 2
            elif rest[i] == "--engine-kind":
                kind = rest[i + 1]; i += 2
            elif rest[i] == "--engine-args":
                full = True; i += 1
            else:
                sys.stderr.write("unknown arg: %s\n" % rest[i])
                return 2
        print(render(root or ".", container, kind, full))
        return 0
    sys.stderr.write(__doc__ or "")
    sys.stderr.write("\nusage: resolved_config.py section --root R --container C "
                     "[--engine-kind K] [--engine-args]\n"
                     "       resolved_config.py env-filter < NAME=VALUE lines\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
