# Documentation for Python Vision

## How to Access Camera WebGUI

### Easy Access (Not Supported Yet?)

Visit the `[CameraName].local:8080/index.html`

For Example:

- Front right camera: [frontrightcamera.local:8080/index.html](http://frontrightcamera.local:8080/index.html)
- Front left camera: [frontleftcamera.local:8080/index.html](http://frontleftcamera.local:8080/index.html)

### Not working? Try this!

Type the IP of the camera and visit index.html on port 8080 (EX: `10.9.48.11:8080/index.html`)

Current WebGUI addresses of cameras:

- Front right camera: [10.9.48.22:8081/index.html](http://10.9.48.22:8081/index.html)
- Front left camera: [10.9.48.22:8080/index.html](http://10.9.48.22:8080/index.html)

## How to set up newly imaged Python Vision

1. Visit [10.9.48.11:8080/index.html](10.9.48.11:8080/index.html)
2. Configure IP suffix to something other than the default
3. Configure other settings as needed...

## Accessing the OrangePi Command Line

You can SSH into the orangepi by going to your terminal and typing `orangepi@[IP address]`

- For example: `orangepi@10.9.48.11`
- The ssh password is by default `orangepi`

### Useful cmd Commands

- `cd /app/cv`
  - Goes to the directory where Python Vision is
- `ip addr show`
  - Shows the IP addresses of the OrangePi
- `journalctl -u vision --follow`
- `journalctl -u photonvision.service --follow`
- `sudo systemctl restart photonvision.service`

## Common Problems

- "IP address of the pi is unknown!"
  - The default address is x.x.x.11. Otherwise, it should be an address that the team has statically configured.
  - If stuck, link the pi to a monitor and keyboard. Type `ip addr show` to find the IP.
- "The WebGUI is not loading!"
  - Try pinging the IP address in the terminal to see if the IP is valid
  - Try using http instead of https
- "I can't run `sudo` commands without admin password!"
  - The password by default is `orangepi`
