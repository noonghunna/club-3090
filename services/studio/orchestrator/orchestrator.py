#!/usr/bin/env python3
"""Studio orchestrator — long-clip (>15 s, 60 s+) extend-chaining for the OWUI Studio pipe.

The OWUI pipe can't run ffmpeg or read ComfyUI's output dir, so multi-segment chaining
lives here (host-side: ffmpeg + output access). Same proven technique as
services/studio/extend_chain.py: seg1 text->video; seg2..N image->video each conditioned
on the PREVIOUS segment's last frame; ffmpeg-concat into one clip written into ComfyUI's
output dir (so the gallery serves it).

Async job API:
  POST /extend  {prompt, lane, segments, frames}  -> {job_id, segments}
  GET  /job/<id>                                  -> {status, progress, filename, subfolder, error}
  GET  /health                                    -> {ok: true}
"""
import json, re, time, subprocess, urllib.request, os, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY = os.environ.get("COMFYUI_URL", "http://localhost:8188")
OUTROOT = os.environ.get("COMFYUI_OUTPUT_DIR", "/output")          # mounted ComfyUI output dir
VIDEODIR = os.path.join(OUTROOT, "video")
STUDIO_PIPE = os.environ.get("STUDIO_PIPE", "/studio/studio_pipe.py")
PORT = int(os.environ.get("PORT", "8190"))
MAX_SEGMENTS = int(os.environ.get("MAX_SEGMENTS", "12"))           # ~120 s cap
SEG_TIMEOUT = int(os.environ.get("SEG_TIMEOUT", "2400"))

# Lane workflows are defined once in the pipe; read them from there (mounted ro).
WF = json.loads(re.search(r'WORKFLOWS = json\.loads\(r"""(.*?)"""\)',
                          open(STUDIO_PIPE).read(), re.S).group(1))
JOBS = {}

def _submit(wf):
    req = urllib.request.Request(COMFY + "/prompt",
        data=json.dumps({"prompt": wf, "client_id": "studio-orch"}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    if r.get("node_errors"):
        raise RuntimeError("comfy node_errors " + json.dumps(r["node_errors"])[:300])
    return r["prompt_id"]

def _wait(pid):
    t0 = time.time()
    while time.time() - t0 < SEG_TIMEOUT:
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
                raise RuntimeError("comfy gen error")
    raise TimeoutError("render timeout")

def _last_frame(mp4, out_png):
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of",
                        "default=nokey=1:noprint_wrappers=1", mp4], capture_output=True, text=True).stdout.strip()
    idx = max(0, int(n) - 1)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf", f"select=eq(n\\,{idx})",
                    "-vframes", "1", "-y", out_png], check=True)
    return out_png

def _upload(png):
    raw = open(png, "rb").read(); fn = os.path.basename(png); bnd = "----orchbnd"
    body = (b"--" + bnd.encode() + b"\r\n"
            b'Content-Disposition: form-data; name="image"; filename="' + fn.encode() + b'"\r\n'
            b"Content-Type: image/png\r\n\r\n" + raw + b"\r\n"
            b"--" + bnd.encode() + b"\r\n"
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            b"--" + bnd.encode() + b"--\r\n")
    req = urllib.request.Request(COMFY + "/upload/image", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    return json.load(urllib.request.urlopen(req, timeout=60)).get("name", fn)

def _run(jid, prompt, lane, segments, frames):
    j = JOBS[jid]
    try:
        segs = []
        for i in range(segments):
            j["progress"] = f"{i+1}/{segments}"
            if i == 0:
                wf = json.loads(json.dumps(WF[lane + "-t2v"]))
                wf["5"]["inputs"]["text"] = prompt; wf["10"]["inputs"]["value"] = frames
            else:
                png = _last_frame(os.path.join(VIDEODIR, segs[-1]), f"/tmp/orch_lf_{jid}_{i}.png")
                wf = json.loads(json.dumps(WF[lane + "-i2v"]))
                wf["5"]["inputs"]["text"] = prompt; wf["10"]["inputs"]["value"] = frames
                wf["100"]["inputs"]["image"] = _upload(png)
            fn = _wait(_submit(wf))
            if not fn:
                raise RuntimeError(f"segment {i+1} produced no output")
            segs.append(fn)
        lst = f"/tmp/orch_concat_{jid}.txt"
        with open(lst, "w") as f:
            for fn in segs:
                f.write("file '%s'\n" % os.path.join(VIDEODIR, fn))
        out = f"studio_combined_{jid[:8]}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y",
                        os.path.join(VIDEODIR, out)], check=True)
        j.update(status="done", filename=out, subfolder="video", segments_done=segs)
    except Exception as e:
        j.update(status="error", error=str(e)[:300])

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "lanes": list(WF)})
        m = re.match(r"/job/([0-9a-f-]+)$", self.path)
        if m:
            j = JOBS.get(m.group(1))
            return self._send(200 if j else 404, j or {"error": "no such job"})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/extend":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        d = json.loads(self.rfile.read(n) or b"{}")
        prompt = (d.get("prompt") or "").strip()
        lane = d.get("lane", "sulphur"); lane = lane if lane in ("sulphur", "ltx") else "sulphur"
        segments = max(2, min(int(d.get("segments", 2)), MAX_SEGMENTS))
        frames = int(d.get("frames", 241))
        if not prompt:
            return self._send(400, {"error": "prompt required"})
        jid = str(uuid.uuid4())
        JOBS[jid] = {"status": "running", "progress": f"0/{segments}", "filename": None,
                     "subfolder": "video", "error": None, "segments": segments}
        threading.Thread(target=_run, args=(jid, prompt, lane, segments, frames), daemon=True).start()
        self._send(200, {"job_id": jid, "segments": segments})
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print(f"[orchestrator] :{PORT} comfy={COMFY} out={VIDEODIR} lanes={list(WF)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
