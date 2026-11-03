# llamacpp-spec.sh — read a llama.cpp container's drafter: is one configured, and what
# did it accept? Sourced (no shebang) by verify-full.sh (step 9), health.sh and report.sh.
# Stdlib only: docker, grep, awk, sed.
#
# Two sources, each chosen because the obvious alternative is unreliable:
#
# - CONFIGURED comes from llama-server's own command line (`docker top` runs ps on the
#   host, so the image needs no tools), not the startup log, whose wording differs
#   between builds: PrismML prints "speculative decoding enabled: draft-mtp", mainline
#   at the v0.6.0 pin printed neither that nor "speculative decoding context
#   initialized". Only the llama-server process counts: a compose's wrapper shell
#   carries its whole script, comments included, as its argument (the moe-cache
#   Flash-Next one contains "-md path.").
# - WHAT IT DID comes from the per-request line llama-server prints in its timing block
#   whenever it drafted (mainline, PrismML and our moe-cache fork, which appends
#   ", mean len = N"; MTP, draft models, DFlash and n-gram all count through the same
#   stats):
#       draft acceptance = 0.95833 (  184 accepted /   192 generated)

# llamacpp_spec_args <container> — the drafting flags on llama-server's command line,
# space-joined, or nothing. `--spec-type none` is not a drafter.
# Dot decimals whatever the user's locale: under es_ES / de_DE mawk read "1.65" as 1 and bash printf rejected "9.0" (#1926, #159).
# LC_ALL would override LC_NUMERIC, so its value moves to the other categories first. Guarded by test-numeric-locale.sh.
[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C
llamacpp_spec_args() {
  docker top "$1" -eo pid,args 2>/dev/null \
    | awk 'NR > 1 && $2 ~ /(^|\/)llama-server$/ { $1 = ""; print; exit }' \
    | command grep -oE -- '(--spec-type[ =][^ ]+|--spec-draft-model[ =][^ ]+|--model-draft[ =][^ ]+|(^| )-md [^ ]+)' \
    | command grep -vE -- '--spec-type[ =]none$' | sed 's/^ //' | tr '\n' ' ' | sed 's/ *$//' || true
}

# llamacpp_draft_rate — sum the draft lines on stdin. Prints
# "<rate> <accepted> <generated> <requests>", or nothing when no line drafted anything.
# Summed, not averaged: a 4-token draft and a 2,900-token one don't weigh the same.
llamacpp_draft_rate() {
  command grep -oE 'draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\)' \
    | awk '{ gsub(/[()]/, " "); a += $5; g += $8; n++ } END { if (g > 0) printf "%.3f %d %d %d\n", a / g, a, g, n }' || true
}
