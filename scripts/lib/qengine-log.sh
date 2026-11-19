# qengine-log.sh — QEngine's own log lines, read in one place (health.sh, report.sh).
#
# QEngine prints llama.cpp-style `[stats]` timing lines, vLLM-style `SpecDecoding metrics` lines, a
# `[chat]` line per request and, on Flash-Next, an expert-cache line; its boot ends with
# `graph warm-up:` on both models. Each reader takes the log text on stdin and prints nothing when the
# log has no such line (an idle server, or a release that words it differently), so a caller can say
# "not in the log" rather than print an empty value. Sourced; defines functions only.

# Numbers parse and print with a dot whatever the locale: mawk under es_ES prints 3.52 as "3,52" (#1926).
# LC_ALL would override LC_NUMERIC, so its value moves to the other categories first. Guarded by test-numeric-locale.sh.
[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C

# qengine_spec_lines — the last 5 SpecDecoding lines, one-step lines left out: a 1-token request
# (AL 1.00) is a warm-up, not a rate.
qengine_spec_lines() {
  command grep -E '\[stats\] SpecDecoding metrics:' | command grep -vE ', 1 steps\)' | tail -5 || true
}

# qengine_spec_al — the mean acceptance length over the lines on stdin (2 decimals), or nothing.
qengine_spec_al() {
  command grep -oE 'Mean acceptance length: [0-9.]+' | awk '{ s += $4; n++ } END { if (n) printf "%.2f", s/n }' || true
}

# qengine_spec_split — the last line's draft split, "MTP a/b, lookup c/d" (Flash-Next drafts mostly
# by lookup, so the acceptance length alone hides which drafter did the work), or nothing.
qengine_spec_split() {
  tail -1 | command grep -oE 'MTP [0-9]+/[0-9]+, lookup [0-9]+/[0-9]+' || true
}

# qengine_expert_cache — Flash-Next's latest cold-group hit rate, decode and prefill; the 27B has no
# expert cache, so nothing.
qengine_expert_cache() {
  command grep -oE 'cold-group hit rate [0-9.]+% decode \([0-9/]+\) [0-9.]+% prefill' | tail -1 || true
}

# qengine_boot_block [max] — the boot: every line up to and including the first `graph warm-up:`,
# at most `max` lines (default 200). A boot that never got there, a crash, shows its first `max`
# lines. On a short QEngine boot (55-65 lines) a plain `head -200` is mostly per-request lines, which
# is what pushed #2103's report past GitHub's issue size limit.
qengine_boot_block() {
  awk -v max="${1:-200}" 'NR > max { exit } { print } /^graph warm-up:/ { exit }'
}

# qengine_request_lines [n] — the last `n` per-request lines (default 3). QEngine logs one per request
# as `[<kind>] slot N prompt …`: `[chat]` for chat completions, `[messages]` for the Anthropic Messages
# API (1.3.6). Matched by that shape, not by kind, so a new endpoint's lines are not missed.
qengine_request_lines() {
  command grep -E '^\[[a-z]+\] slot [0-9]+ prompt ' | tail -n "${1:-3}" || true
}

# qengine_runtime_tail [n] — the last `n` lines (default 40) logged after the boot, the per-request
# lines and the `[stats]` lines left out (the summary reads those): warnings, VRAM pressure, archive
# and cache messages. Nothing when the boot never finished, since the boot block already holds it all.
qengine_runtime_tail() {
  awk '/^graph warm-up:/ { booted = 1; next }
       booted && !/^\[stats\] / && !/^\[[a-z]+\] slot [0-9]+ prompt / { print }' | tail -n "${1:-40}"
}
