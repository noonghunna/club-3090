"""Bounded hardware scenarios for profile/compose compatibility checks."""

CANONICAL_SCENARIOS = [
    {"name": "2x-3090-pcie", "hardware": ["rtx-3090", "rtx-3090"], "nvlink_active": False},
    {"name": "2x-3090-nvlink", "hardware": ["rtx-3090", "rtx-3090"], "nvlink_active": True},
    {"name": "4x-3090-pcie", "hardware": ["rtx-3090", "rtx-3090", "rtx-3090", "rtx-3090"], "nvlink_active": False},
    # Added 2026-08-15 with vllm/qwen38-27b-multi8-max. ⚠️ This list must COVER every
    # `tp` value present in COMPOSE_REGISTRY or the self-test in test-profiles-compat
    # silently stops checking the largest topologies: it reports "no fitting canonical
    # scenario" for a slug that is actually fine, because no scenario has enough cards
    # to evaluate it. `test-canonical-scenarios-cover-registry-tp` now asserts that.
    # NOTE: diagnose_profile_cli.py does NOT read this list — it SYNTHESISES a scenario
    # from the entry's own world size (f"{n}x-{hardware_id}-pcie"), which is why it
    # passed multi8 while this list failed it. That split is deliberate (one asks "does
    # it fit some real host", the other "does it fit the host it implies"), so the two
    # are kept consistent by the coverage test rather than by merging them.
    {"name": "8x-3090-pcie", "hardware": ["rtx-3090"] * 8, "nvlink_active": False},
    {"name": "1x-3090", "hardware": ["rtx-3090"], "nvlink_active": False},
    {"name": "2x-4090", "hardware": ["rtx-4090", "rtx-4090"], "nvlink_active": False},
    {"name": "1x-5090", "hardware": ["rtx-5090"], "nvlink_active": False},
    {"name": "2x-5090", "hardware": ["rtx-5090", "rtx-5090"], "nvlink_active": False},
    {"name": "2x-a100-40gb", "hardware": ["a100-40gb", "a100-40gb"], "nvlink_active": True},
    {"name": "heterogeneous-3090-5090", "hardware": ["rtx-3090", "rtx-5090"], "nvlink_active": False},
]
