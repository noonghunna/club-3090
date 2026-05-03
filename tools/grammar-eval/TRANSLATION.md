# Holiday tagline grammar translation for vLLM/xgrammar

This translates Holiday_Purpose_3166's Reddit GBNF grammar into the form used
by club-3090's bounded-thinking vLLM stack.

The translated grammar lives at:

```text
tools/grammar-eval/holiday-tagline.gbnf
```

## Translated grammar

```gbnf
root ::= "Q=" q "\n" "M=" m "\n" "K=" toks "\n" "R=" toks "\n" "V=" v "\n" "</think>\n\n" out

q ::= "solve" | "prove" | "route" | "debug" | "patch" | "code" | "calc" | "compare" | "explain"
m ::= "case" | "enum" | "check" | "derive" | "edit" | "test" | "trace" | "rank"
v ::= "ok" | "fail" | "done" | "blocked" | "candidate" | "verify"

toks ::= tok | tok "," tok | tok "," tok "," tok | tok "," tok "," tok "," tok | tok "," tok "," tok "," tok "," tok
tok ::= alpha tail18
alpha ::= [A-Za-z]
tok_char ::= [A-Za-z0-9_.!-]

tail18 ::= "" | tok_char tail17
tail17 ::= "" | tok_char tail16
tail16 ::= "" | tok_char tail15
tail15 ::= "" | tok_char tail14
tail14 ::= "" | tok_char tail13
tail13 ::= "" | tok_char tail12
tail12 ::= "" | tok_char tail11
tail11 ::= "" | tok_char tail10
tail10 ::= "" | tok_char tail9
tail9 ::= "" | tok_char tail8
tail8 ::= "" | tok_char tail7
tail7 ::= "" | tok_char tail6
tail6 ::= "" | tok_char tail5
tail5 ::= "" | tok_char tail4
tail4 ::= "" | tok_char tail3
tail3 ::= "" | tok_char tail2
tail2 ::= "" | tok_char tail1
tail1 ::= "" | tok_char tail0
tail0 ::= ""

out ::= [\x09\x0A\x0D\x20-\x7E]+
```

## Translation decisions

- **Opening `<think>\n` removed:** Qwen3.6's chat template already prefixes
  the assistant reasoning channel with `<think>\n`. The grammar therefore
  starts at `Q=`, matching the existing `fsm_grammar_no_open.gbnf` port.

- **Closing `</think>\n\n` preserved verbatim:** the grammar still forces the
  model to terminate the reasoning block and then emit normal answer text.
  vLLM's Qwen reasoning parser may expose only the body as `reasoning_content`;
  the smoke test normalizes this before regex validation.

- **`tok` bounded repetition unrolled:** Holiday's original
  `tok ::= [A-Za-z][A-Za-z0-9_.!-]{0,18}` was rewritten as `alpha tail18`
  with explicit nullable tail rules. This preserves the same 1-19 character
  token length without relying on `{0,18}` support.

- **Keyword lists preserved:** `q`, `m`, `v`, and the 1-5 comma-separated
  `toks` alternatives are unchanged except for formatting.

- **ASCII answer region preserved:** `out ::= [\x09\x0A\x0D\x20-\x7E]+`
  matches the current structured-CoT grammar's permissive code/output region.
  It allows tabs, newlines, carriage returns, spaces, and printable ASCII.

## Validation status

The local host environment does not have `xgrammar` installed, so static parser
validation must happen through vLLM. Use:

```bash
python3 tools/grammar-eval/smoke-test.py --boot
```

The smoke test sends the grammar through
`structured_outputs: {"grammar": ...}` and verifies that generated reasoning
matches:

```regex
^Q=\w+\nM=\w+\nK=[\w,.!-]+\nR=[\w,.!-]+\nV=\w+$
```

Any 4xx/5xx response or shape mismatch should be treated as a translation
failure before running the 30-prompt subset bench.

## Phase-1 smoke results — 2026-05-03 PM

Ran against `vllm/bounded-thinking` on Qwen3.6-27B AutoRound INT4, vLLM
`0.20.1rc1.dev16+g7a1eb8ac2`, Genesis `2db18df`/v7.69, RTX 3090. All five
tagline-grammar prompts passed with shape-matching output:

```
[grammar-smoke] PASS simple-math/tagline: ok
[grammar-smoke] PASS simple-code/tagline: ok
[grammar-smoke] PASS simple-debug/tagline: ok
[grammar-smoke] PASS fallback-hard-1/tagline: ok
[grammar-smoke] PASS fallback-hard-2/tagline: ok
```

Three FREE-baseline prompts (run alongside for sanity comparison) failed
with `empty answer content` — that's the existing `max_tokens=4096`
truncation trap documented in [`docs/STRUCTURED_COT.md`](../../docs/STRUCTURED_COT.md)
"Honest caveats" section, not a grammar issue. Tagline grammar's structure
caps think tokens, dodges the trap.

**Translation status:** validated. Phase 2 (30-prompt HumanEval+ subset
bench) is unblocked.
