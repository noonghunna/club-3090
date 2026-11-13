# Blender headless + blender-mcp addon socket server.
# The addon (ahujasid/blender-mcp v1.8) exposes a JSON command socket on 9876
# (get_scene_info, execute_code, asset search, …). An MCP bridge (blender-mcp
# on PyPI) connects agents to that socket; this image serves the Blender side.
FROM ubuntu:24.04

ARG BLENDER_VERSION=4.2.9
# sha256 of blender-${BLENDER_VERSION}-linux-x64.tar.xz, from download.blender.org's
# blender-${BLENDER_VERSION}.sha256. Change both together.
ARG BLENDER_SHA256=dfbc127a7d28f9c2175b23bf9d6701b2855f31eedfb391f9a6e60adb24572846
# The addon is pinned to a commit, not `main`: upstream has no tags, and `main` moved to
# addon v2.0 after this image was validated on v1.8. Bump REF and SHA256 together.
ARG BLENDER_MCP_REF=7001b2845a5673984de660b98f1ea9ba3b95f0a5
ARG BLENDER_MCP_SHA256=a7b3873aed8405bd2785292ab9b916a4392129f82a02a336b62ac14aa121828b
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils bzip2 \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
      libgl1 libegl1 libgl1-mesa-dri libsm6 libice6 libgomp1 xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

# Blender LTS, unpacked under /opt/blender. The bundled python is self-contained.
# Tarball layout: blender-X/4.2/scripts/addons + 4.2/python — note the nested 4.2.
RUN curl --retry 5 --retry-delay 3 --retry-all-errors -fsSL "https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
      -o /tmp/blender.tar.xz \
    && echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c - \
    && tar -xf /tmp/blender.tar.xz -C /opt \
    && mkdir -p /opt/blender/${BLENDER_VERSION} \
    && mv /opt/blender-${BLENDER_VERSION}-linux-x64/* /opt/blender/${BLENDER_VERSION}/ \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/${BLENDER_VERSION}/blender /usr/local/bin/blender

# blender-mcp addon (v1.8, pinned above) into the install's addons tree + requests for
# its asset-library features (PolyHaven/Meshy/Tripo) in Blender's bundled python.
RUN mkdir -p /opt/blender/${BLENDER_VERSION}/4.2/scripts/addons/blendermcp \
    && curl --retry 5 --retry-delay 3 --retry-all-errors -fsSL \
      "https://raw.githubusercontent.com/ahujasid/blender-mcp/${BLENDER_MCP_REF}/addon.py" \
      -o /opt/blender/${BLENDER_VERSION}/4.2/scripts/addons/blendermcp/__init__.py \
    && echo "${BLENDER_MCP_SHA256}  /opt/blender/${BLENDER_VERSION}/4.2/scripts/addons/blendermcp/__init__.py" | sha256sum -c -
RUN /opt/blender/${BLENDER_VERSION}/4.2/python/bin/python3.11 -m pip install --no-cache-dir requests
# Blender 4.2 no longer scans the bundled scripts/addons — user addons live in
# the user-config scripts dir, which IS on the addon scan path.
RUN mkdir -p /root/.config/blender/4.2/scripts/addons/blendermcp \
 && cp /opt/blender/${BLENDER_VERSION}/4.2/scripts/addons/blendermcp/__init__.py \
      /root/.config/blender/4.2/scripts/addons/blendermcp/__init__.py

# Enable the addon, point its bind address at 0.0.0.0 (the default pref binds
# localhost, which is unreachable from outside the container), then start the
# socket server. Blender stays in the foreground; the server runs on its timer.
COPY start-mcp-server.py /opt/start-mcp-server.py
# blender --background is refused by the blender-mcp addon (commands would never
# execute); a virtual display gives a real (software-GL) GUI context instead.
# xvfb-run’s readiness loop can hang before exec’ing the client — launch Xvfb
# and blender directly. No auth needed: display is loopback-only in-container.
CMD ["bash", "-c", "Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp & sleep 3; export DISPLAY=:99; stdbuf -oL -eL blender --python /opt/start-mcp-server.py"]

EXPOSE 9876

# bash exists in the base image; probe the addon socket for liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD bash -c 'echo > /dev/tcp/127.0.0.1/9876' || exit 1
