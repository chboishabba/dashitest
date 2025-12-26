from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from vulkan import *  # type: ignore
except Exception:
    vkmod = None
else:
    import vulkan as vkmod


def _run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return ""
    except subprocess.CalledProcessError as exc:
        return exc.stdout + exc.stderr
    return proc.stdout


def _print(title: str, body: str) -> None:
    print(f"\n== {title} ==")
    if body.strip():
        print(body.rstrip())
    else:
        print("(no output)")


def probe_ffmpeg() -> None:
    out = _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    _print("ffmpeg hwaccels", out)


def probe_vaapi_device(path: str) -> None:
    print("\n== vaapi device ==")
    dev = Path(path)
    if dev.exists():
        print(f"found {dev}")
    else:
        print(f"missing {dev}")
    if Path("/dev/dri").exists():
        print("/dev/dri entries:")
        for entry in sorted(Path("/dev/dri").iterdir()):
            print(f"- {entry}")


def probe_vulkan() -> None:
    print("\n== vulkan extensions ==")
    if vkmod is None:
        print("python-vulkan not available")
        return
    try:
        inst_exts = vkEnumerateInstanceExtensionProperties(None)
        inst_names = sorted(ext.extensionName for ext in inst_exts)
        print("instance extensions:")
        for name in inst_names:
            if "external" in name.lower() or "dma" in name.lower() or "drm" in name.lower():
                print(f"* {name}")
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="vaapi_probe",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="none",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 0),
        )
        inst_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        instance = vkCreateInstance(inst_info, None)
        devices = vkEnumeratePhysicalDevices(instance)
        if not devices:
            print("no physical devices")
            vkDestroyInstance(instance, None)
            return
        for idx, dev in enumerate(devices):
            props = vkGetPhysicalDeviceProperties(dev)
            print(f"\nphysical device {idx}: {props.deviceName}")
            dev_exts = vkEnumerateDeviceExtensionProperties(dev, None)
            dev_names = sorted(ext.extensionName for ext in dev_exts)
            for name in dev_names:
                if "external" in name.lower() or "dma" in name.lower() or "drm" in name.lower():
                    print(f"* {name}")
        vkDestroyInstance(instance, None)
    except Exception as exc:
        print(f"vulkan probe failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe VAAPI + Vulkan external memory support.")
    parser.add_argument("--vaapi-device", default="/dev/dri/renderD128")
    args = parser.parse_args()

    probe_ffmpeg()
    probe_vaapi_device(args.vaapi_device)
    probe_vulkan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
