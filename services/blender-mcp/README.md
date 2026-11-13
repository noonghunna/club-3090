# blender-mcp service (services/blender-mcp)

Headless-Blender-with-virtual-display + blender-mcp addon socket, shared as one
MCP server by **omp**, **pi**, and **dsh**.

## Architecture

- `ahujasid/blender-mcp` addon (v1.8, pinned to commit `7001b284` with a sha256
  check — upstream has no tags and `main` has moved on to v2.0) runs **inside**
  Blender and exposes a JSON command socket on **9876** (`get_scene_info`,
  `execute_code`, PolyHaven asset search/import, …). The addon **refuses
  `blender -b`** (commands would never execute), so the container runs Blender in
  **GUI mode on an `Xvfb` virtual display** with software GL (llvmpipe). It starts
  `Xvfb` directly rather than through `xvfb-run`, whose readiness wait can hang.
  `allow_remote_connections` + `bind_address 0.0.0.0` are set programmatically at
  startup (`start-mcp-server.py`).
- Blender 4.2 no longer scans the bundled `scripts/addons` — the addon is copied
  into the **user** addons dir (`/root/.config/blender/4.2/scripts/addons/`).
- Agents spawn their own **MCP bridge** (`blender-mcp` on PyPI, stdio) pointed at
  the container socket: `uvx blender-mcp --host 127.0.0.1 --port 9876`. The
  addon serves clients **sequentially** — one agent drives Blender at a time;
  a second agent's bridge connects but blocks until the first disconnects.
- Scene + `.blend` files live in the `blender-data` volume; container rebuilds
  keep the scene.

## Run

```bash
cd services/blender-mcp && docker compose up -d --build   # port 127.0.0.1:9876
docker logs -f blender-mcp                                # socket server status
```

## Agent wiring

| Agent | Config | Server entry |
|---|---|---|
| omp | `~/.claude.json` → `mcpServers.blender` (stdio) | `uvx blender-mcp --host 127.0.0.1 --port 9876` |
| pi | `~/.pi/agent/mcp.json` → `mcpServers.blender` | same command |
| dsh | `~/.dsh/profiles/web/cordis.patch.yml` → `mcp-client-blender` patch (`@deepseek-ai/dsh-mcp-client`, stdio) | same; `toolCallTimeoutMs: 600000` (scene ops run tens of seconds) |

omp/pi pick the config up on next session start; dsh on next `dsh web` boot.
`~/.claude.json` is shared with the Claude CLI — edit with python json round-trip,
not sed.

## Caveats

- **`execute_code` runs any Python a client sends**, inside Blender, as root in
  the container. The socket is published on `127.0.0.1` only; don't publish it
  wider.
- A long render blocks the addon's command queue: other agents' calls wait until
  it finishes.
- The bridge is versioned separately from the addon: this addon (v1.8) was
  validated with the PyPI bridge `blender-mcp` 2.0.0, which is what `uvx
  blender-mcp` resolves to today. To hold that pairing, use `uvx blender-mcp@2.0.0`.
- The addon needs `requests` inside Blender's **bundled** python (installed in
  the Dockerfile; Blender 4.2 bundles python 3.11).
- A stop that doesn't let Xvfb clean up (a Docker daemon restart, a kill) leaves its
  lock in the container; the image's start command removes it, so the container comes
  back instead of restart-looping on "Server is already active for display 99".
- GPU rendering (Cycles CUDA/OptiX) is unavailable in this container by design
  (software GL); Eevee/Cycles renders go through llvmpipe (slow but works).
