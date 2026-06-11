#!/usr/bin/env python3
"""Extend a clip past the single-pass ceiling by chaining segments (Phase B, host-side).

seg1 = text->video; seg2..N = image->video conditioned on the PREVIOUS segment's last
frame; then ffmpeg-concat into one video. Requires host ffmpeg + read access to ComfyUI's
output dir, so it runs on the host (not inside the OWUI pipe). Validated on 2x 3090: joins
are seamless for slow/ambient scenes (fast action may show a slight velocity reset at a
cut — native LTX temporal-extend would smooth that; future work). See docs/ai-studio/video.md.

Env: COMFYUI_URL (default http://localhost:8188) · COMFYUI_OUTPUT_DIR (default
/mnt/models/comfyui/output). Reads the workflows from the sibling studio_pipe.py.

Usage: python3 extend_chain.py "<prompt>" <n_segments> <frames_per_seg>
"""
import json, re, time, subprocess, urllib.request, sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.environ.get("COMFYUI_URL", "http://localhost:8188")
OUTDIR = os.path.join(os.environ.get("COMFYUI_OUTPUT_DIR", "/mnt/models/comfyui/output"), "video")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else (
    "A serene mountain lake at dawn, mist drifting over still water, a slow continuous "
    "cinematic dolly forward across the surface, soft ambient birdsong and gentle wind.")
NSEG = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 241  # 10s/seg (crisp zone)

WF = json.loads(re.search(r'WORKFLOWS = json\.loads\(r"""(.*?)"""\)',
                          open(os.path.join(_HERE, "studio_pipe.py")).read(), re.S).group(1))

def submit(wf):
    req = urllib.request.Request(COMFY + "/prompt",
        data=json.dumps({"prompt": wf, "client_id": "extend"}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    if r.get("node_errors"):
        raise RuntimeError("node_errors " + json.dumps(r["node_errors"])[:300])
    return r["prompt_id"]

def wait(pid, tmo=1800):
    t0 = time.time()
    while time.time() - t0 < tmo:
        time.sleep(5)
        h = json.load(urllib.request.urlopen(COMFY + "/history/" + pid, timeout=30))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                for node in h[pid].get("outputs", {}).values():
                    for v in (node.get("gifs") or node.get("videos") or node.get("images") or []):
                        if str(v.get("filename", "")).endswith(".mp4"):
                            return v["filename"]
                return None
            if st.get("status_str") == "error":
                raise RuntimeError("gen error: " + json.dumps(st.get("messages", []))[-300:])
    raise TimeoutError("render timed out")

def last_frame(mp4, out_png):
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of",
                        "default=nokey=1:noprint_wrappers=1", mp4],
                       capture_output=True, text=True).stdout.strip()
    idx = max(0, int(n) - 1)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf",
                    f"select=eq(n\\,{idx})", "-vframes", "1", "-y", out_png], check=True)
    return out_png

def upload(png):
    raw = open(png, "rb").read(); fn = os.path.basename(png); bnd = "----extbnd9"
    body = (b"--" + bnd.encode() + b"\r\n"
            b'Content-Disposition: form-data; name="image"; filename="' + fn.encode() + b'"\r\n'
            b"Content-Type: image/png\r\n\r\n" + raw + b"\r\n"
            b"--" + bnd.encode() + b"\r\n"
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            b"--" + bnd.encode() + b"--\r\n")
    req = urllib.request.Request(COMFY + "/upload/image", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    return json.load(urllib.request.urlopen(req, timeout=60)).get("name", fn)

print(f"[ext] chaining {NSEG} x {FRAMES}f (~{NSEG*FRAMES/24:.0f}s) — sulphur lane", flush=True)
segs = []
for i in range(NSEG):
    if i == 0:
        wf = json.loads(json.dumps(WF["sulphur-t2v"]))
        wf["5"]["inputs"]["text"] = PROMPT; wf["10"]["inputs"]["value"] = FRAMES
        kind = "t2v"
    else:
        png = last_frame(os.path.join(OUTDIR, segs[-1]), f"/tmp/ext_lf_{i}.png")
        name = upload(png)
        wf = json.loads(json.dumps(WF["sulphur-i2v"]))
        wf["5"]["inputs"]["text"] = PROMPT; wf["10"]["inputs"]["value"] = FRAMES
        wf["100"]["inputs"]["image"] = name
        kind = "i2v<-lastframe"
    print(f"[ext] seg {i+1}/{NSEG} ({kind}) rendering...", flush=True)
    t0 = time.time(); fn = wait(submit(wf))
    print(f"  -> {fn} ({time.time()-t0:.0f}s)", flush=True)
    segs.append(fn)

listfile = "/tmp/ext_concat.txt"
with open(listfile, "w") as f:
    for fn in segs:
        f.write("file '%s'\n" % os.path.join(OUTDIR, fn))
combined = f"/tmp/ext_combined_{NSEG}x{FRAMES}.mp4"
subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", combined], check=False)
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nokey=1:noprint_wrappers=1", combined],
                     capture_output=True, text=True).stdout.strip()
print(f"[ext] COMBINED -> {combined}  duration={dur}s  segments={segs}", flush=True)
