"""#1153 Route-K — `derive_compose_facts` reads a user-supplied compose.

The BYOM user most likely to exist has weights on disk and a working compose, and
until Route-K the funnel had no door for them: ① takes an HF repo, and the only
compose that could enter was one c3 itself emitted.

This is the pure read half, tested here without a TUI (the same seam
test_family_scaffold.py uses for the promote scaffold). It must handle the three
shapes a compose actually takes in the wild — YAML list, folded string, and JSON
exec form — because a per-flag regex gets the list form wrong: it captures the
next line's `-` instead of the value.
"""

from __future__ import annotations

from club3090_cockpit.data import derive_compose_facts


VLLM_LIST = """# Profile (at-a-glance):
#   Status: 🧪 Experimental
services:
  vllm:
    image: vllm/vllm-openai:v0.27.1
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    ports:
      - "${PORT:-8101}:8000"
    command:
      - --model
      - /models/my-model
      - --served-model-name
      - my-model
      - --max-model-len
      - "131072"
      - --kv-cache-dtype
      - fp8_e4m3
      - --tensor-parallel-size
      - "2"
"""

LLAMA_FOLDED = """services:
  llama:
    image: ghcr.io/ggerganov/llama.cpp:server
    ports:
      - "9001:8080"
    command: >-
      -m /models/mine.gguf -c 65536 -ctk q8_0 -a mine -ts 2
"""

EXEC_FORM = """services:
  vllm:
    image: vllm/vllm-openai:v0.27.1
    command: ["--model=/w/x", "--max-model-len=32768", "--tensor-parallel-size=2"]
"""


class TestDeriveComposeFacts:
    def test_vllm_yaml_list_form(self):
        f = derive_compose_facts(VLLM_LIST, path="/tmp/x.yml")
        assert f.ok and not f.error
        assert f.engine == "vllm"
        assert f.model_path == "/models/my-model"
        assert f.served_name == "my-model"
        assert f.max_ctx == "131072"
        assert f.kv_dtype == "fp8_e4m3"
        assert f.tp == "2"
        assert f.port == "8101"          # ${PORT:-8101}, not the container 8000
        assert "Experimental" in f.status_header
        assert f.path == "/tmp/x.yml"

    def test_llama_folded_string_form(self):
        f = derive_compose_facts(LLAMA_FOLDED)
        assert f.ok
        assert f.engine == "llama-cpp"
        assert f.model_path == "/models/mine.gguf"
        assert f.max_ctx == "65536"
        assert f.kv_dtype == "q8_0"
        assert f.served_name == "mine"
        assert f.tp == "2"
        assert f.port == "9001"          # from the ports: mapping

    def test_json_exec_form(self):
        f = derive_compose_facts(EXEC_FORM)
        assert f.ok
        assert f.model_path == "/w/x"    # --flag=value
        assert f.max_ctx == "32768"
        assert f.tp == "2"

    def test_a_flag_never_captures_the_next_flag_as_its_value(self):
        """The bug a per-flag regex makes: `- --model` followed by `- --foo`."""
        f = derive_compose_facts(
            "services:\n  v:\n    image: vllm/vllm-openai:v0.27.1\n"
            "    command:\n      - --model\n      - --trust-remote-code\n"
        )
        assert f.model_path == "", f"captured a flag as a value: {f.model_path!r}"

    def test_tp_falls_back_to_the_device_count(self):
        f = derive_compose_facts(
            "services:\n  v:\n    image: vllm/vllm-openai:v0.27.1\n"
            "    environment:\n      - CUDA_VISIBLE_DEVICES=0,1,2,3\n"
        )
        assert f.tp == "4"

    def test_empty_and_non_compose_are_refused_not_guessed(self):
        assert derive_compose_facts("").error == "empty compose"
        assert derive_compose_facts("   \n").ok is False
        f = derive_compose_facts("services:\n  x:\n    build: .\n")
        assert f.ok is False and "no image" in f.error

    def test_never_raises_on_junk(self):
        for junk in ("\x00\x01", "]]]{{{", "image:\n" * 500, "- " * 1000):
            derive_compose_facts(junk)   # must not raise
