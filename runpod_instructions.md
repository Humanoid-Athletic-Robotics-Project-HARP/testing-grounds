1. Setup a pod on runpod.io. Deploy with Humanoid-Dev-GUI
2. Once done, spin up an interactive web terminal. Run:

```
echo -e "123123\n123123\n\n" | vncpasswd
vncserver :1 -geometry 1920x1080 -depth 24
websockify --web=/usr/share/novnc 5999 localhost:5901 &
```

3. Open up the VNC port (5999) and click `vnc.html`.

A good test is running `isaacsim` in the VNC terminal on.

Enjoy!