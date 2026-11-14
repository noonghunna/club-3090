# Enable blender-mcp, bind 0.0.0.0 (addon default pref is localhost, which the
# host cannot reach), start the JSON socket server, keep Blender foreground.
import addon_utils
import bpy

print("[blender-mcp] enabling addon...", flush=True)
addon_utils.enable("blendermcp", default_set=True, persistent=True)

prefs = bpy.context.preferences.addons.get("blendermcp")
if prefs is not None:
    prefs.preferences.bind_address = "0.0.0.0"
    prefs.preferences.allow_remote_connections = True
    print("[blender-mcp] bind_address set to 0.0.0.0 (remote connections allowed)", flush=True)
else:
    raise RuntimeError("blendermcp addon did not register; check the addons path")

scene = bpy.context.scene
scene.blendermcp_port = 9876
scene.blendermcp_auto_start_server = True

bpy.ops.blendermcp.start_server()
print("[blender-mcp] socket server start operator executed; blender idling", flush=True)

# GUI mode (xvfb): the script ends, blender enters its main event loop, and the
# addon’s command-queue timer fires on the main thread — do NOT block here.
