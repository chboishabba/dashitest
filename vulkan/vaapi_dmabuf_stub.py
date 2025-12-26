from __future__ import annotations

import argparse
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from vulkan import *  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"python-vulkan is required: {exc}") from exc


MAX_OBJECTS = 4
MAX_PLANES = 4

HEADER_FMT = "<IIIII"
OBJ_FMT = "<QQ"
PLANE_FMT = "<III"
INFO_SIZE = (
    struct.calcsize(HEADER_FMT)
    + MAX_OBJECTS * struct.calcsize(OBJ_FMT)
    + MAX_PLANES * struct.calcsize(PLANE_FMT)
)

def _fourcc(a: str, b: str, c: str, d: str) -> int:
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


def _fourcc_str(code: int) -> str:
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


DRM_FORMAT_NV12 = _fourcc("N", "V", "1", "2")
DRM_FORMAT_P010 = _fourcc("P", "0", "1", "0")
DRM_FORMAT_R16 = _fourcc("R", "1", "6", " ")
DRM_FORMAT_MOD_INVALID = 0xFFFFFFFFFFFFFFFF


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout.strip()


def _build_helper(source: Path, binary: Path) -> None:
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return
    cflags = _run(["pkg-config", "--cflags", "libavformat", "libavcodec", "libavutil"]).split()
    libs = _run(["pkg-config", "--libs", "libavformat", "libavcodec", "libavutil"]).split()
    cmd = [
        "cc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-o",
        str(binary),
        str(source),
    ] + cflags + libs
    subprocess.run(cmd, check=True)


def _recv_dmabuf(sock: socket.socket) -> Tuple[dict, List[int]]:
    data, ancdata, _, _ = sock.recvmsg(INFO_SIZE, socket.CMSG_SPACE(MAX_OBJECTS * 4))
    if len(data) < INFO_SIZE:
        raise RuntimeError("short dmabuf metadata read")
    fds: List[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.extend(struct.unpack(f"<{len(cmsg_data) // 4}i", cmsg_data))
    offset = 0
    width, height, drm_format, nb_objects, nb_planes = struct.unpack_from(HEADER_FMT, data, offset)
    offset += struct.calcsize(HEADER_FMT)
    objects = []
    for _ in range(MAX_OBJECTS):
        modifier, size = struct.unpack_from(OBJ_FMT, data, offset)
        objects.append({"modifier": modifier, "size": size})
        offset += struct.calcsize(OBJ_FMT)
    planes = []
    for _ in range(MAX_PLANES):
        obj_index, plane_offset, pitch = struct.unpack_from(PLANE_FMT, data, offset)
        planes.append({"object_index": obj_index, "offset": plane_offset, "pitch": pitch})
        offset += struct.calcsize(PLANE_FMT)
    return (
        {
            "width": width,
            "height": height,
            "drm_format": drm_format,
            "nb_objects": nb_objects,
            "nb_planes": nb_planes,
            "objects": objects,
            "planes": planes,
        },
        fds,
    )


def _choose_queue_family(physical_device) -> int:
    families = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for idx, fam in enumerate(families):
        if fam.queueFlags & VK_QUEUE_GRAPHICS_BIT:
            return idx
    for idx, fam in enumerate(families):
        if fam.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return idx
    return 0


def _find_memory_type(physical_device, type_bits: int) -> int:
    props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    for i in range(props.memoryTypeCount):
        if type_bits & (1 << i):
            if props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                return i
    for i in range(props.memoryTypeCount):
        if type_bits & (1 << i):
            return i
    raise RuntimeError("no compatible memory type found")


def _import_dmabuf_image(info: dict, fds: List[int]) -> None:
    if info["nb_objects"] != 1:
        raise RuntimeError(f"expected 1 dmabuf object, got {info['nb_objects']}")
    if any(plane["object_index"] != 0 for plane in info["planes"][: info["nb_planes"]]):
        raise RuntimeError("multi-object plane layout not supported")

    vk_p010 = globals().get("VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM")
    if vk_p010 is None:
        vk_p010 = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM

    format_map = {
        DRM_FORMAT_NV12: (VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, 2),
        DRM_FORMAT_P010: (vk_p010, 2),
        DRM_FORMAT_R16: (VK_FORMAT_R16_UNORM, 1),
    }
    fmt_entry = format_map.get(info["drm_format"])
    if fmt_entry is None:
        fourcc = _fourcc_str(info["drm_format"])
        raise RuntimeError(
            f"unsupported drm format: 0x{info['drm_format']:08x} ('{fourcc}')"
        )
    vk_format, expected_planes = fmt_entry
    if info["nb_planes"] != expected_planes:
        if info["drm_format"] == DRM_FORMAT_R16 and info["nb_planes"] == 2:
            vk_format = vk_p010
            expected_planes = 2
        else:
            raise RuntimeError(
                f"unexpected plane count {info['nb_planes']} for format 0x{info['drm_format']:08x}"
            )

    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vaapi_dmabuf_stub",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 0, 0),
    )
    instance = vkCreateInstance(VkInstanceCreateInfo(sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info), None)
    devices = vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise RuntimeError("no Vulkan physical devices found")
    physical_device = devices[0]
    props = vkGetPhysicalDeviceProperties(physical_device)

    required_exts = {
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_EXT_external_memory_dma_buf",
    }
    dev_exts = vkEnumerateDeviceExtensionProperties(physical_device, None)
    dev_names = {ext.extensionName for ext in dev_exts}
    has_modifier_ext = "VK_EXT_image_drm_format_modifier" in dev_names
    missing = sorted(required_exts - dev_names)
    if missing:
        related = sorted(name for name in dev_names if any(key in name.lower() for key in ("external", "dma", "drm")))
        hint = "try setting VK_ICD_FILENAMES to a driver that exposes VK_EXT_image_drm_format_modifier"
        raise RuntimeError(
            "missing Vulkan device extensions: "
            f"{', '.join(missing)} (device={props.deviceName}; related={related}; {hint})"
        )

    modifier = info["objects"][0]["modifier"]
    if not has_modifier_ext and modifier not in (0, DRM_FORMAT_MOD_INVALID):
        raise RuntimeError(
            "dmabuf uses non-linear DRM modifier "
            f"0x{modifier:x}; VK_EXT_image_drm_format_modifier is required"
        )
    if modifier == DRM_FORMAT_MOD_INVALID:
        modifier = 0

    queue_family_index = _choose_queue_family(physical_device)
    queue_info = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    device = vkCreateDevice(
        physical_device,
        VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(required_exts) + (1 if has_modifier_ext else 0),
            ppEnabledExtensionNames=list(required_exts) + (["VK_EXT_image_drm_format_modifier"] if has_modifier_ext else []),
        ),
        None,
    )
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    external_info = VkExternalMemoryImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    )
    image_tiling = VK_IMAGE_TILING_LINEAR
    image_pnext = external_info
    if has_modifier_ext:
        plane_layouts = [
            VkSubresourceLayout(
                offset=info["planes"][idx]["offset"],
                rowPitch=info["planes"][idx]["pitch"],
                size=0,
                arrayPitch=0,
                depthPitch=0,
            )
            for idx in range(info["nb_planes"])
        ]
        drm_info = VkImageDrmFormatModifierExplicitCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
            drmFormatModifier=modifier,
            drmFormatModifierPlaneCount=info["nb_planes"],
            pPlaneLayouts=plane_layouts,
            pNext=external_info,
        )
        image_tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
        image_pnext = drm_info
    image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        pNext=image_pnext,
        imageType=VK_IMAGE_TYPE_2D,
        format=vk_format,
        extent=VkExtent3D(width=info["width"], height=info["height"], depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=image_tiling,
        usage=VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    image = vkCreateImage(device, image_info, None)

    mem_req = vkGetImageMemoryRequirements(device, image)
    memory_type = _find_memory_type(physical_device, mem_req.memoryTypeBits)
    alloc_size = max(mem_req.size, info["objects"][0]["size"])
    import_info = VkImportMemoryFdInfoKHR(
        sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        fd=fds[0],
    )
    alloc_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=alloc_size,
        memoryTypeIndex=memory_type,
        pNext=import_info,
    )
    memory = vkAllocateMemory(device, alloc_info, None)
    vkBindImageMemory(device, image, memory, 0)

    command_pool = vkCreateCommandPool(
        device,
        VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
        ),
        None,
    )
    cmd = vkAllocateCommandBuffers(
        device,
        VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ),
    )[0]
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
    aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT
    if expected_planes == 2:
        aspect_mask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT
    barrier = VkImageMemoryBarrier(
        sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        srcAccessMask=0,
        dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
        oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        newLayout=VK_IMAGE_LAYOUT_GENERAL,
        image=image,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=aspect_mask,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        None,
        0,
        None,
        1,
        [barrier],
    )
    vkEndCommandBuffer(cmd)
    submit = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[cmd],
    )
    vkQueueSubmit(queue, 1, [submit], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)

    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyImage(device, image, None)
    vkFreeMemory(device, memory, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal VAAPI dmabuf -> Vulkan import stub.")
    parser.add_argument("video", type=Path, help="Path to input video.")
    parser.add_argument("--vaapi-device", default="/dev/dri/renderD128")
    args = parser.parse_args()

    source = Path(__file__).with_name("vaapi_dmabuf_export.c")
    binary = Path(__file__).with_name("vaapi_dmabuf_export")
    _build_helper(source, binary)

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    env = os.environ.copy()
    env["DMABUF_STUB_SOCK_FD"] = str(child_sock.fileno())
    proc = subprocess.Popen(
        [str(binary), str(args.video), args.vaapi_device],
        pass_fds=[child_sock.fileno()],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_sock.close()

    info, fds = _recv_dmabuf(parent_sock)
    parent_sock.close()
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(out + err)

    fds = fds[: info["nb_objects"]]
    if len(fds) < info["nb_objects"]:
        raise RuntimeError("did not receive enough dmabuf fds")

    _import_dmabuf_image(info, fds)
    print(
        f"imported dmabuf frame {info['width']}x{info['height']} "
        f"format=0x{info['drm_format']:08x} modifier=0x{info['objects'][0]['modifier']:x}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
