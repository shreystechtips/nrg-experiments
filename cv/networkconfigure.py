import argparse
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

# Script to monitor ip_suffix.txt for changes and update network configuration accordingly.
# Assumes /24 subnet mask, avahi-daemon for mDNS (.local), and that the connection "Orange Pi ethernet" exists.
# Runs indefinitely, polling every 1 second for quick detection of changes.
# Creates a lock file during processing to prevent concurrent modifications.
parser = argparse.ArgumentParser(
    prog="NetworkConfigure", description="Sync Python vision to latest IP and hostname."
)

parser.add_argument("--file", default=Path(__file__).parent / "ip_suffix.txt")
args = parser.parse_args()
FILE = args.file
LOCK = f"{FILE}.lock"
CON_NAME = None
SLEEP_INTERVAL = 1  # Poll every 1 second for quick detection of changes

last_mtime = 0
last_gateway = ""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("network-configure")


def get_ethernet_interfaces() -> List[Dict[str, str]]:
    """
    Returns a list of all ethernet/wired interfaces managed by NetworkManager.

    Each item in the list is a dictionary with keys:
        - name:       connection name (e.g. "Orange Pi ethernet", "Wired connection 1")
        - device:     interface name (e.g. "eth0", "enp3s0", "enx001122334455")
        - type:       usually "ethernet"
        - uuid:       connection UUID
        - state:      connection state (e.g. "connected", "disconnected")

    Returns empty list if nmcli fails or no ethernet connections exist.
    """
    try:
        # Get all connections in a machine-readable format
        result = subprocess.run(
            [
                "nmcli",
                "-t",
                "--fields",
                "NAME,UUID,TYPE,DEVICE,STATE",
                "connection",
                "show",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        interfaces = []

        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue

            fields = line.split(":")
            if len(fields) < 5:
                continue

            name, uuid, conn_type, device, state = fields[:5]

            # Only include ethernet type connections
            # (excludes wifi, bridge, bond, vlan, tun, etc.)
            if "ethernet" in conn_type.lower():
                entry = {
                    "name": name,
                    "uuid": uuid,
                    "type": conn_type,
                    "device": device if device else None,
                    "state": state,
                }
                interfaces.append(entry)

        return interfaces

    except subprocess.CalledProcessError as e:
        log.error(f"Error running nmcli: {e}")
        log.error(f"stderr: {e.stderr}")
        return []
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return []


def touch_lock():
    with open(LOCK, "w") as f:
        pass


def remove_lock():
    if os.path.exists(LOCK):
        os.remove(LOCK)


# Try to get the ethernet connection name in nmcli!
while CON_NAME is None:
    detected_ifaces = get_ethernet_interfaces()
    if len(detected_ifaces) == 1:
        CON_NAME = detected_ifaces[0]["name"]
        log.info("UPDATED SYSTEM CONNECTION NAME TO %s", CON_NAME)
    else:
        log.info("NO ETHERNET INTERFACES DETECTED... TRYING AGAIN IN 0.8S")
        time.sleep(0.8)

remove_lock()


def get_gateway():
    try:
        output = subprocess.check_output(["ip", "route", "show"]).decode()
        match = re.search(r"default via ([\d.]+)", output)
        return match.group(1) if match else None
    except Exception:
        return None


def get_current_ips():
    try:
        output = (
            subprocess.check_output(
                ["nmcli", "-g", "IP4.ADDRESS", "con", "show", CON_NAME]
            )
            .decode()
            .strip()
        )
        return (
            {x.strip() for x in output.split("|")} if output else set()
        )  # Take first if multiple
    except Exception:
        return set()


def ping(address, count=1, timeout=1):
    try:
        subprocess.check_call(
            ["ping", "-c", str(count), "-W", str(timeout), address],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def update_network(new_ips, gateway, old_ips, old_gateway):
    try:
        log.info("Setting IP: %s %s %s %s", new_ips, gateway, old_ips, old_gateway)
        if set(new_ips) == set(old_ips):
            log.info("Settings same as before, skipping.")
            return True
        dat = ",".join(new_ips)
        subprocess.check_call(
            [
                "nmcli",
                "con",
                "mod",
                CON_NAME,
                "ipv4.addresses",
                dat,
                # "ipv4.gateway",
                # gateway,
            ]
        )
        subprocess.check_call(["nmcli", "con", "down", CON_NAME])
        subprocess.check_call(["nmcli", "con", "up", CON_NAME])
        return True
        # time.sleep(2)  # Wait for stabilization
        # if ping(gateway, count=1, timeout=2):
        #     log.info(f"Success: New IP {new_ip} configured and verified.")
        #     return True
        # else:
        #     log.info(
        #         f"Error: Connectivity check failed after update. Reverting to {old_ip}."
        #     )
        #     subprocess.check_call(
        #         [
        #             "nmcli",
        #             "con",
        #             "mod",
        #             CON_NAME,
        #             "ipv4.addresses",
        #             old_ip,
        #             "ipv4.gateway",
        #             old_gateway,
        #         ]
        #     )
        #     subprocess.check_call(["nmcli", "con", "down", CON_NAME])
        #     subprocess.check_call(["nmcli", "con", "up", CON_NAME])
        #     return False
    except Exception as e:
        log.info(f"Error during network update: {e}")
        return False


def update_hostname(host):
    try:
        current_host = subprocess.check_output(["hostname"]).decode().strip()
        log.info("%s %s", current_host, host)
        if host and host != current_host:
            subprocess.check_call(["hostnamectl", "set-hostname", host])
            # Check if avahi-daemon is active
            if (
                subprocess.call(
                    ["systemctl", "is-active", "avahi-daemon"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0
            ):
                subprocess.check_call(["systemctl", "restart", "avahi-daemon"])
                log.info(
                    f"Hostname updated to '{host}' and avahi-daemon restarted for .local resolution."
                )
            else:
                log.info(
                    f"Hostname updated to '{host}'. (avahi-daemon not active; .local may not resolve.)"
                )
    except Exception as e:
        log.info(f"Error updating hostname: {e}")


def gen_expected_ips():
    prefixes = ["10.9.48", "192.168.2"]

    # Get current configured IP
    current_ips = get_current_ips()

    # correlated = []
    log.info(current_ips)
    tmp_curr_ips_by_prefix = {".".join(ip.split(".")[:3]): ip for ip in current_ips}
    new_ip_addrs = dict()
    new_ips = set()
    for prefix in prefixes:
        ip = tmp_curr_ips_by_prefix.get(prefix)
        new_ip = f"{prefix}.{suffix}/24"
        new_ip_addrs[prefix] = (ip, new_ip)
        new_ips.add(new_ip)

    return (new_ip_addrs, current_ips, new_ips)


expected_ips = set()

while True:
    if os.path.exists(FILE):
        mtime = os.stat(FILE).st_mtime
        gateway = get_gateway()

        if (
            (gateway and (mtime != last_mtime or gateway != last_gateway))
            or not expected_ips
            or expected_ips != get_current_ips()
        ):
            if not os.path.exists(LOCK):
                touch_lock()

                try:
                    with open(FILE, "r") as f:
                        lines = f.readlines()
                        suffix = lines[0].strip() if lines else ""
                        host = lines[1].strip() if len(lines) > 1 else ""

                    # Validate suffix (10-255, integer)
                    if not suffix.isdigit() or int(suffix) < 10 or int(suffix) > 255:
                        log.info(
                            f"Error: Invalid IP suffix '{suffix}' (must be integer 10-255)."
                        )
                        time.sleep(SLEEP_INTERVAL)
                        continue

                    # Validate host (no spaces)
                    if " " in host:
                        log.info(f"Error: Hostname '{host}' contains spaces.")
                        time.sleep(SLEEP_INTERVAL)
                        continue

                    # Compute new IP based on current gateway's prefix
                    # prefix = ".".join(gateway.split(".")[:3])
                    new_ip_addrs, current_ips, new_ips = gen_expected_ips()

                    log.info("%s %s", new_ip_addrs, current_ips)
                    if new_ips != current_ips:
                        # Check if new IP is in use (ping to detect potential conflict)
                        for prefix, (current_ip, new_ip_addr) in new_ip_addrs.items():
                            if ping(new_ip_addr):
                                log.info(
                                    f"Warning: Potential IP conflict detected for {new_ip_addr} (ping responded). Skipping update."
                                )
                                continue
                        # Backup old config
                        old_ips = [v[0] for v in new_ip_addrs.values()]
                        old_gateway = last_gateway if last_gateway else gateway

                        # Apply new config and verify
                        update_network(
                            [v[1] for v in new_ip_addrs.values()],
                            gateway,
                            old_ips,
                            old_gateway,
                        )

                    # Update hostname if changed
                    update_hostname(host)

                    # Update last known values
                    last_mtime = mtime
                    last_gateway = gateway
                    expected_ips = new_ips

                except Exception as e:
                    log.info(f"Error processing file: {e}")
                finally:
                    remove_lock()

    else:
        log.info(f"Warning: File {FILE} not found. Skipping check.")

    time.sleep(SLEEP_INTERVAL)
