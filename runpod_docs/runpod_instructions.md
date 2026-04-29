# RunPod Setup Instructions

1. Set up a pod on [runpod.io](https://runpod.io). Deploy using the **Humanoid-Dev-GUI** template.

2. Once the pod is running, open an interactive web terminal and run:

```bash
# Set the VNC password non-interactively.
# The four fields are: password, confirm password, view-only password (blank = skip), confirm (blank = skip).
# Change "123123" to something stronger if this pod is long-lived or shared.
echo -e "123123\n123123\n\n" | vncpasswd

# Start a VNC server at display :1 with 1080p resolution.
vncserver :1 -geometry 1920x1080 -depth 24

# Bridge the VNC port to a web-accessible websocket so you can connect via browser.
websockify --web=/usr/share/novnc 5999 localhost:5901 &
```

3. In the RunPod network settings, expose port **5999**, then open the URL and click **`vnc.html`**.

---

## Smoke Tests

A good first check is launching `isaacsim` in the VNC terminal to confirm the GUI and GPU are working.

You can also run a random agent on the cartpole balancing task to verify that physics simulation and 3D geometry load correctly:

```bash
/usr/bin/python3 scripts/environments/random_agent.py \
  --task Isaac-Cartpole-Direct-v0 \
  --num_envs 128
```

Enjoy!
