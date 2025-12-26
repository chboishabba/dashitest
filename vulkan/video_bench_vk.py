from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import glfw  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("glfw is required for preview: pip install glfw") from exc

from vulkan import *
from vulkan import ffi
import vulkan as vkmod

try:
    from .decode_backend import decode_gray_buffer, ffprobe_video
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from decode_backend import decode_gray_buffer, ffprobe_video
try:
    from .dmabuf_export import export_dmabuf
except ImportError:
    from dmabuf_export import export_dmabuf


SHADER_DIR = Path(__file__).resolve().parents[1] / "vulkan_compute" / "shaders"
VK_SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _fourcc(a: str, b: str, c: str, d: str) -> int:
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


DRM_FORMAT_NV12 = _fourcc("N", "V", "1", "2")


def _load_instance_func(instance: VkInstance, name: str, required: bool = True):
    fn = getattr(vkmod, name, None)
    if fn is not None:
        return fn
    try:
        return vkGetInstanceProcAddr(instance, name)
    except (ProcedureNotFoundError, ExtensionNotSupportedError):
        if required:
            raise RuntimeError(f"{name} is unavailable in python-vulkan.")
        return None


def _load_device_func(device: VkDevice, name: str, required: bool = True):
    fn = getattr(vkmod, name, None)
    if fn is not None:
        return fn
    try:
        return vkGetDeviceProcAddr(device, name)
    except (ProcedureNotFoundError, ExtensionNotSupportedError):
        if required:
            raise RuntimeError(f"{name} is unavailable in python-vulkan.")
        return None


def _select_physical_device(instance: VkInstance) -> VkPhysicalDevice:
    try:
        devices = vkEnumeratePhysicalDevices(instance)
    except VkErrorInitializationFailed as exc:
        icd_env = os.environ.get("VK_ICD_FILENAMES")
        icd_paths = glob.glob("/usr/share/vulkan/icd.d/*.json")
        hint = f"VK_ICD_FILENAMES={icd_env}" if icd_env else "VK_ICD_FILENAMES not set"
        raise RuntimeError(
            "vkEnumeratePhysicalDevices failed. "
            f"{hint}. Available ICDs: {icd_paths}"
        ) from exc
    if not devices:
        raise RuntimeError("No Vulkan physical devices found.")
    return devices[0]


def _find_queue_family_index(
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    surface: VkSurfaceKHR,
) -> int:
    props = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    surface_support = _load_instance_func(
        instance, "vkGetPhysicalDeviceSurfaceSupportKHR", required=False
    )
    for idx, prop in enumerate(props):
        if not (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT):
            continue
        if surface_support is not None:
            if surface_support(physical_device, idx, surface):
                return idx
        else:
            if glfw.get_physical_device_presentation_support(instance, physical_device, idx):
                return idx
    raise RuntimeError("No queue family supports graphics+present.")


def _find_memory_type(
    mem_props: VkPhysicalDeviceMemoryProperties,
    type_bits: int,
    required_flags: int,
) -> int:
    for idx in range(mem_props.memoryTypeCount):
        if type_bits & (1 << idx):
            flags = mem_props.memoryTypes[idx].propertyFlags
            if (flags & required_flags) == required_flags:
                return idx
    raise RuntimeError("No compatible memory type found.")


def _read_spirv(path: Path) -> bytes:
    return path.read_bytes()


def _read_spirv_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _ensure_spirv(spv_path: Path, src_path: Path) -> None:
    if not src_path.exists():
        raise SystemExit(f"Missing shader source: {src_path}")
    if spv_path.exists():
        try:
            if spv_path.stat().st_mtime >= src_path.stat().st_mtime:
                return
        except OSError:
            pass
    try:
        subprocess.run(["glslc", str(src_path), "-o", str(spv_path)], check=True)
    except FileNotFoundError as exc:
        raise SystemExit("glslc is required to compile shaders.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Vulkan video bench preview (GPU path).")
    parser.add_argument("video", type=Path, help="Path to input video (e.g., MP4).")
    parser.add_argument("--frames", type=int, default=240, help="Max frames to decode for the preview.")
    parser.add_argument("--vaapi", action="store_true", help="Use VAAPI hw decode if available.")
    parser.add_argument(
        "--vaapi-zero-copy",
        action="store_true",
        help="Validate zero-copy prerequisites (dmabuf import path not yet implemented).",
    )
    parser.add_argument(
        "--vaapi-device",
        default="/dev/dri/renderD128",
        help="VAAPI device path.",
    )
    parser.add_argument(
        "--vaapi-dmabuf",
        action="store_true",
        help="Use VAAPI dmabuf export + GPU NV12/P010->RGBA conversion (single frame).",
    )
    parser.add_argument("--dmabuf-timeout", type=float, default=5.0, help="Seconds to wait for dmabuf export.")
    parser.add_argument(
        "--dmabuf-force-linear",
        action="store_true",
        help="Force a linear dmabuf via download/upload (requires /dev/dri/card*).",
    )
    parser.add_argument(
        "--dmabuf-drm-device",
        default=None,
        help="DRM device for dmabuf force-linear uploads (defaults to --vaapi-device).",
    )
    parser.add_argument("--dmabuf-debug", action="store_true", help="Print dmabuf exporter logs.")
    parser.add_argument(
        "--mode",
        choices=("raw", "diff"),
        default="raw",
        help="Display mode: raw frame or diff vs previous frame.",
    )
    parser.add_argument("--stats-every", type=int, default=60, help="Print stats every N frames.")
    parser.add_argument("--in-flight", type=int, default=2, help="Max frames in flight.")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    if args.vaapi_zero_copy:
        raise SystemExit(
            "--vaapi-zero-copy is not implemented yet. Run `python vulkan/vaapi_probe.py` "
            "to verify prerequisites, then use --vaapi for now."
        )

    use_dmabuf = args.vaapi_dmabuf
    dmabuf_info = None
    dmabuf_fds = None
    frames = None
    if use_dmabuf:
        if args.dmabuf_force_linear:
            drm_device = args.dmabuf_drm_device or args.vaapi_device
            if not os.path.exists(drm_device):
                raise SystemExit(
                    f"--dmabuf-force-linear requested but --dmabuf-drm-device '{drm_device}' does not exist.\n"
                    "Expose /dev/dri/card* or disable --dmabuf-force-linear."
                )
        dmabuf_info, dmabuf_fds, log = export_dmabuf(
            args.video,
            args.vaapi_device,
            force_linear=args.dmabuf_force_linear,
            drm_device=args.dmabuf_drm_device,
            timeout_s=args.dmabuf_timeout,
            debug=args.dmabuf_debug,
        )
        if args.dmabuf_debug and log:
            print(log)
        width = dmabuf_info["width"]
        height = dmabuf_info["height"]
    else:
        width, height, nb_frames = ffprobe_video(args.video)
        max_frames = args.frames
        if nb_frames:
            max_frames = min(max_frames, nb_frames)
        decode_start = time.perf_counter()
        frames, used_vaapi = decode_gray_buffer(
            args.video, width, height, max_frames, args.vaapi, args.vaapi_device
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        if frames.size == 0:
            raise SystemExit("No frames decoded.")
        print(
            f"decoded {frames.shape[0]} frames {width}x{height} "
            f"via {'vaapi' if used_vaapi else 'cpu'} in {decode_ms:.1f} ms"
        )

    if not glfw.init():
        raise SystemExit("Failed to initialize GLFW.")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(width, height, "Vulkan Video Bench", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Failed to create GLFW window.")

    required_exts = glfw.get_required_instance_extensions()
    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vulkan_video_bench",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 0, 0),
    )
    instance_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
        enabledExtensionCount=len(required_exts),
        ppEnabledExtensionNames=required_exts,
    )
    instance = vkCreateInstance(instance_info, None)

    surface_ptr = ffi.new("VkSurfaceKHR*")
    result = glfw.create_window_surface(instance, window, None, surface_ptr)
    if result != VK_SUCCESS:
        raise RuntimeError(f"glfwCreateWindowSurface failed with VkResult={result}.")
    surface = surface_ptr[0]

    physical_device = _select_physical_device(instance)
    queue_family_index = _find_queue_family_index(instance, physical_device, surface)

    queue_priority = 1.0
    queue_info = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=[queue_priority],
    )
    device_info = VkDeviceCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_info],
        enabledExtensionCount=1,
        ppEnabledExtensionNames=[VK_KHR_SWAPCHAIN_EXTENSION_NAME],
    )
    device = vkCreateDevice(physical_device, device_info, None)
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    destroy_surface = _load_instance_func(instance, "vkDestroySurfaceKHR")
    surface_caps_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
    surface_formats_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
    present_modes_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR")

    surface_caps = surface_caps_fn(physical_device, surface)
    formats = list(surface_formats_fn(physical_device, surface))
    present_modes = list(present_modes_fn(physical_device, surface))
    surface_format = formats[0]
    for fmt in formats:
        if fmt.format == VK_FORMAT_B8G8R8A8_UNORM:
            surface_format = fmt
            break
    present_mode = VK_PRESENT_MODE_FIFO_KHR
    if VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
        present_mode = VK_PRESENT_MODE_MAILBOX_KHR
    extent = surface_caps.currentExtent
    swap_width = extent.width if extent.width != 0xFFFFFFFF else width
    swap_height = extent.height if extent.height != 0xFFFFFFFF else height

    image_count = max(surface_caps.minImageCount + 1, 2)
    if surface_caps.maxImageCount > 0:
        image_count = min(image_count, surface_caps.maxImageCount)

    swapchain_info = VkSwapchainCreateInfoKHR(
        sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        surface=surface,
        minImageCount=image_count,
        imageFormat=surface_format.format,
        imageColorSpace=surface_format.colorSpace,
        imageExtent=VkExtent2D(swap_width, swap_height),
        imageArrayLayers=1,
        imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
        preTransform=surface_caps.currentTransform,
        compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        presentMode=present_mode,
        clipped=VK_TRUE,
        oldSwapchain=VK_NULL_HANDLE,
    )
    create_swapchain = _load_device_func(device, "vkCreateSwapchainKHR")
    get_swapchain_images = _load_device_func(device, "vkGetSwapchainImagesKHR")
    acquire_next_image = _load_device_func(device, "vkAcquireNextImageKHR")
    queue_present = _load_device_func(device, "vkQueuePresentKHR")
    destroy_swapchain = _load_device_func(device, "vkDestroySwapchainKHR")

    swapchain = create_swapchain(device, swapchain_info, None)
    swap_images = list(get_swapchain_images(device, swapchain))
    swap_views = []
    for image in swap_images:
        view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=surface_format.format,
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        swap_views.append(vkCreateImageView(device, view_info, None))

    mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    curr_image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    prev_image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    out_image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    curr_image = vkCreateImage(device, curr_image_info, None)
    prev_image = vkCreateImage(device, prev_image_info, None)
    out_image = vkCreateImage(device, out_image_info, None)

    curr_reqs = vkGetImageMemoryRequirements(device, curr_image)
    prev_reqs = vkGetImageMemoryRequirements(device, prev_image)
    out_reqs = vkGetImageMemoryRequirements(device, out_image)
    curr_type = _find_memory_type(
        mem_props,
        curr_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    prev_type = _find_memory_type(
        mem_props,
        prev_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    out_type = _find_memory_type(
        mem_props,
        out_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    curr_alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=curr_reqs.size,
        memoryTypeIndex=curr_type,
    )
    prev_alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=prev_reqs.size,
        memoryTypeIndex=prev_type,
    )
    out_alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=out_reqs.size,
        memoryTypeIndex=out_type,
    )
    curr_memory = vkAllocateMemory(device, curr_alloc, None)
    prev_memory = vkAllocateMemory(device, prev_alloc, None)
    out_memory = vkAllocateMemory(device, out_alloc, None)
    vkBindImageMemory(device, curr_image, curr_memory, 0)
    vkBindImageMemory(device, prev_image, prev_memory, 0)
    vkBindImageMemory(device, out_image, out_memory, 0)
    curr_view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=curr_image,
        viewType=VK_IMAGE_VIEW_TYPE_2D,
        format=VK_FORMAT_R8_UNORM,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    prev_view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=prev_image,
        viewType=VK_IMAGE_VIEW_TYPE_2D,
        format=VK_FORMAT_R8_UNORM,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    out_view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=out_image,
        viewType=VK_IMAGE_VIEW_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    curr_view = vkCreateImageView(device, curr_view_info, None)
    prev_view = vkCreateImageView(device, prev_view_info, None)
    out_view = vkCreateImageView(device, out_view_info, None)

    sampler_info = VkSamplerCreateInfo(
        sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        magFilter=VK_FILTER_NEAREST,
        minFilter=VK_FILTER_NEAREST,
        addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    )
    sampler = vkCreateSampler(device, sampler_info, None)

    vert_src = SHADER_DIR / "preview.vert"
    frag_src = SHADER_DIR / "preview.frag"
    vert_shader = SHADER_DIR / "preview.vert.spv"
    frag_shader = SHADER_DIR / "preview.frag.spv"
    _ensure_spirv(vert_shader, vert_src)
    _ensure_spirv(frag_shader, frag_src)

    vert_bytes = _read_spirv_bytes(vert_shader)
    frag_bytes = _read_spirv_bytes(frag_shader)
    vert_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(vert_bytes),
            pCode=vert_bytes,
        ),
        None,
    )
    frag_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(frag_bytes),
            pCode=frag_bytes,
        ),
        None,
    )

    diff_src = VK_SHADER_DIR / "diff.comp"
    diff_shader = VK_SHADER_DIR / "diff.spv"
    _ensure_spirv(diff_shader, diff_src)
    diff_bytes = _read_spirv_bytes(diff_shader)
    diff_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(diff_bytes),
            pCode=diff_bytes,
        ),
        None,
    )

    nv12_src = VK_SHADER_DIR / "nv12_to_rgba.comp"
    nv12_shader = VK_SHADER_DIR / "nv12_to_rgba.spv"
    _ensure_spirv(nv12_shader, nv12_src)
    nv12_bytes = _read_spirv_bytes(nv12_shader)
    nv12_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(nv12_bytes),
            pCode=nv12_bytes,
        ),
        None,
    )

    compute_set_layout = vkCreateDescriptorSetLayout(
        device,
        VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=3,
            pBindings=[
                VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                VkDescriptorSetLayoutBinding(
                    binding=2,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ],
        ),
        None,
    )
    compute_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=ffi.new("VkDescriptorSetLayout[]", [compute_set_layout]),
            pushConstantRangeCount=1,
            pPushConstantRanges=[VkPushConstantRange(stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=4)],
        ),
        None,
    )
    compute_entry = ffi.new("char[]", b"main")
    compute_stage = VkPipelineShaderStageCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=diff_module,
        pName=compute_entry,
    )
    compute_ci = VkComputePipelineCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=compute_stage,
        layout=compute_layout,
    )
    compute_ci_arr = ffi.new("VkComputePipelineCreateInfo[]", [compute_ci])
    compute_pipeline = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        compute_ci_arr,
        None,
    )[0]

    nv12_set_layout = vkCreateDescriptorSetLayout(
        device,
        VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=2,
            pBindings=[
                VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
                VkDescriptorSetLayoutBinding(
                    binding=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ],
        ),
        None,
    )
    nv12_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=ffi.new("VkDescriptorSetLayout[]", [nv12_set_layout]),
            pushConstantRangeCount=1,
            pPushConstantRanges=[VkPushConstantRange(stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=28)],
        ),
        None,
    )
    nv12_stage = VkPipelineShaderStageCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=nv12_module,
        pName=compute_entry,
    )
    nv12_ci = VkComputePipelineCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=nv12_stage,
        layout=nv12_layout,
    )
    nv12_pipeline = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        ffi.new("VkComputePipelineCreateInfo[]", [nv12_ci]),
        None,
    )[0]

    graphics_set_layout = vkCreateDescriptorSetLayout(
        device,
        VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=[
                VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
                )
            ],
        ),
        None,
    )
    graphics_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=ffi.new("VkDescriptorSetLayout[]", [graphics_set_layout]),
        ),
        None,
    )

    color_attachment = VkAttachmentDescription(
        format=surface_format.format,
        samples=VK_SAMPLE_COUNT_1_BIT,
        loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
        storeOp=VK_ATTACHMENT_STORE_OP_STORE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    )
    color_ref = VkAttachmentReference(attachment=0, layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    subpass = VkSubpassDescription(
        pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
        colorAttachmentCount=1,
        pColorAttachments=[color_ref],
    )
    render_pass = vkCreateRenderPass(
        device,
        VkRenderPassCreateInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_attachment],
            subpassCount=1,
            pSubpasses=[subpass],
        ),
        None,
    )

    vert_entry = ffi.new("char[]", b"main")
    frag_entry = ffi.new("char[]", b"main")
    shader_stages = ffi.new(
        "VkPipelineShaderStageCreateInfo[]",
        [
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_VERTEX_BIT,
                module=vert_module,
                pName=vert_entry,
            ),
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag_module,
                pName=frag_entry,
            ),
        ],
    )
    viewports = ffi.new("VkViewport[]", [VkViewport(0, 0, swap_width, swap_height, 0.0, 1.0)])
    scissors = ffi.new("VkRect2D[]", [VkRect2D(VkOffset2D(0, 0), VkExtent2D(swap_width, swap_height))])
    color_attachments = ffi.new(
        "VkPipelineColorBlendAttachmentState[]",
        [
            VkPipelineColorBlendAttachmentState(
                colorWriteMask=VK_COLOR_COMPONENT_R_BIT
                | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT
            )
        ],
    )
    pipeline = vkCreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE,
        1,
        [
            VkGraphicsPipelineCreateInfo(
                sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=2,
                pStages=shader_stages,
                pVertexInputState=VkPipelineVertexInputStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                ),
                pInputAssemblyState=VkPipelineInputAssemblyStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                ),
                pViewportState=VkPipelineViewportStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    viewportCount=1,
                    pViewports=viewports,
                    scissorCount=1,
                    pScissors=scissors,
                ),
                pRasterizationState=VkPipelineRasterizationStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    polygonMode=VK_POLYGON_MODE_FILL,
                    cullMode=VK_CULL_MODE_NONE,
                    frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    lineWidth=1.0,
                ),
                pMultisampleState=VkPipelineMultisampleStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
                ),
                pColorBlendState=VkPipelineColorBlendStateCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    attachmentCount=1,
                    pAttachments=color_attachments,
                ),
                layout=graphics_layout,
                renderPass=render_pass,
                subpass=0,
            )
        ],
        None,
    )[0]

    descriptor_pool = vkCreateDescriptorPool(
        device,
        VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=3,
            pPoolSizes=[
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1),
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=4),
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1),
            ],
            maxSets=3,
        ),
        None,
    )
    compute_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[compute_set_layout],
        ),
    )[0]
    graphics_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[graphics_set_layout],
        ),
    )[0]
    nv12_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[nv12_set_layout],
        ),
    )[0]
    vkUpdateDescriptorSets(
        device,
        1,
        [
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=graphics_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[
                    VkDescriptorImageInfo(
                        sampler=sampler,
                        imageView=out_view,
                        imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )
                ],
            )
        ],
        0,
        None,
    )
    vkUpdateDescriptorSets(
        device,
        3,
        [
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[VkDescriptorImageInfo(imageView=curr_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[VkDescriptorImageInfo(imageView=prev_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=2,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[VkDescriptorImageInfo(imageView=out_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
            ),
        ],
        0,
        None,
    )

    framebuffers = []
    for view in swap_views:
        framebuffers.append(
            vkCreateFramebuffer(
                device,
                VkFramebufferCreateInfo(
                    sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    renderPass=render_pass,
                    attachmentCount=1,
                    pAttachments=[view],
                    width=swap_width,
                    height=swap_height,
                    layers=1,
                ),
                None,
            )
        )

    max_frames_in_flight = max(1, args.in_flight)
    staging_buffers = []
    staging_mems = []
    staging_maps = []
    dmabuf_buffer = None
    dmabuf_memory = None
    dmabuf_info_obj = None
    if use_dmabuf:
        dmabuf_info_obj = dmabuf_info
        dmabuf_size = int(dmabuf_info_obj["objects"][0]["size"])
        dmabuf_buffer = vkCreateBuffer(
            device,
            VkBufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=dmabuf_size,
                usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        dmabuf_reqs = vkGetBufferMemoryRequirements(device, dmabuf_buffer)
        dmabuf_type = _find_memory_type(mem_props, dmabuf_reqs.memoryTypeBits)
        dmabuf_import = VkImportMemoryFdInfoKHR(
            sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            fd=dmabuf_fds[0],
        )
        dmabuf_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=max(dmabuf_reqs.size, dmabuf_size),
            memoryTypeIndex=dmabuf_type,
            pNext=dmabuf_import,
        )
        dmabuf_memory = vkAllocateMemory(device, dmabuf_alloc, None)
        vkBindBufferMemory(device, dmabuf_buffer, dmabuf_memory, 0)
        vkUpdateDescriptorSets(
            device,
            2,
            [
                VkWriteDescriptorSet(
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=nv12_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[
                        VkDescriptorBufferInfo(buffer=dmabuf_buffer, offset=0, range=dmabuf_size)
                    ],
                ),
                VkWriteDescriptorSet(
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=nv12_set,
                    dstBinding=1,
                    descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[VkDescriptorImageInfo(imageView=out_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
                ),
            ],
            0,
            None,
        )
    else:
        staging_size = width * height
        staging_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=staging_size,
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        for _ in range(max_frames_in_flight):
            staging_buffers.append(vkCreateBuffer(device, staging_info, None))
        staging_reqs = vkGetBufferMemoryRequirements(device, staging_buffers[0])
        staging_type = _find_memory_type(
            mem_props,
            staging_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        staging_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=staging_reqs.size,
            memoryTypeIndex=staging_type,
        )
        for buf in staging_buffers:
            mem = vkAllocateMemory(device, staging_alloc, None)
            vkBindBufferMemory(device, buf, mem, 0)
            mapped = vkMapMemory(device, mem, 0, staging_size, 0)
            staging_mems.append(mem)
            staging_maps.append(np.frombuffer(mapped, dtype=np.uint8, count=staging_size))

    command_pool = vkCreateCommandPool(
        device,
        VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
        ),
        None,
    )
    command_buffers = vkAllocateCommandBuffers(
        device,
        VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(swap_images),
        ),
    )

    image_available = []
    render_done = []
    in_flight = []
    for _ in range(max_frames_in_flight):
        image_available.append(
            vkCreateSemaphore(device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None)
        )
        render_done.append(
            vkCreateSemaphore(device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None)
        )
        in_flight.append(
            vkCreateFence(
                device,
                VkFenceCreateInfo(
                    sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    flags=VK_FENCE_CREATE_SIGNALED_BIT,
                ),
                None,
            )
        )

    frames_flat = None if use_dmabuf else frames.reshape(frames.shape[0], -1)
    first_frame = True
    frame = 0
    start = time.time()
    cpu_prep_ms = 0.0
    cpu_wait_ms = 0.0
    cpu_copy_ms = 0.0
    cpu_record_ms = 0.0
    gpu_submit_ms = 0.0
    cpu_total_ms = 0.0
    total_frames = 1 if use_dmabuf else frames.shape[0]
    while not glfw.window_should_close(window) and frame < total_frames:
        frame_start = time.perf_counter()
        wait_start = time.perf_counter()
        glfw.poll_events()
        frame_idx = frame % max_frames_in_flight
        vkWaitForFences(device, 1, [in_flight[frame_idx]], VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        vkResetFences(device, 1, [in_flight[frame_idx]])
        image_index = acquire_next_image(
            device, swapchain, 0xFFFFFFFFFFFFFFFF, image_available[frame_idx], VK_NULL_HANDLE
        )
        cpu_wait_ms += (time.perf_counter() - wait_start) * 1000.0

        cmd = command_buffers[image_index]
        record_start = time.perf_counter()
        vkResetCommandBuffer(cmd, 0)
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))

        if use_dmabuf:
            copy_ms = 0.0
        else:
            cpu_start = time.perf_counter()
            gray = frames_flat[frame]
            curr_buf = staging_maps[frame_idx]
            curr_buf[:] = gray
            copy_ms = (time.perf_counter() - cpu_start) * 1000.0
            cpu_prep_ms += copy_ms
            cpu_copy_ms += copy_ms

        if use_dmabuf:
            out_to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0 if first_frame else VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED if first_frame else VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=out_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
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
                [out_to_general],
            )
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nv12_pipeline)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nv12_layout, 0, 1, [nv12_set], 0, None)
            fmt_id = 0 if dmabuf_info_obj["drm_format"] == DRM_FORMAT_NV12 else 1
            pc = ffi.new(
                "uint32_t[]",
                [
                    width,
                    height,
                    dmabuf_info_obj["planes"][0]["offset"],
                    dmabuf_info_obj["planes"][1]["offset"],
                    dmabuf_info_obj["planes"][0]["pitch"],
                    dmabuf_info_obj["planes"][1]["pitch"],
                    fmt_id,
                ],
            )
            vkCmdPushConstants(cmd, nv12_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, pc)
            vkCmdDispatch(cmd, (width + 15) // 16, (height + 15) // 16, 1)
            out_to_sample = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                image=out_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [out_to_sample],
            )
        else:
            curr_to_transfer = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED if first_frame else VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                image=curr_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [curr_to_transfer],
            )

            copy_region = VkBufferImageCopy(
                bufferOffset=0,
                bufferRowLength=0,
                bufferImageHeight=0,
                imageSubresource=VkImageSubresourceLayers(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                imageOffset=VkOffset3D(0, 0, 0),
                imageExtent=VkExtent3D(width=width, height=height, depth=1),
            )
            vkCmdCopyBufferToImage(
                cmd,
                staging_buffers[frame_idx],
                curr_image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [copy_region],
            )

            curr_to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=curr_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            out_to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED if first_frame else VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=out_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                2,
                [curr_to_general, out_to_general],
            )
            if first_frame:
                prev_to_transfer = VkImageMemoryBarrier(
                    sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    image=prev_image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                )
                curr_to_src = VkImageMemoryBarrier(
                    sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                    oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                    newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    image=curr_image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                )
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0,
                    None,
                    0,
                    None,
                    2,
                    [prev_to_transfer, curr_to_src],
                )
                vkCmdCopyImage(
                    cmd,
                    curr_image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    prev_image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    [VkImageCopy(
                        srcSubresource=VkImageSubresourceLayers(
                            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                        srcOffset=VkOffset3D(0, 0, 0),
                        dstSubresource=VkImageSubresourceLayers(
                            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                        dstOffset=VkOffset3D(0, 0, 0),
                        extent=VkExtent3D(width=width, height=height, depth=1),
                    )],
                )
                prev_to_general = VkImageMemoryBarrier(
                    sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=VK_IMAGE_LAYOUT_GENERAL,
                    image=prev_image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                )
                curr_back_to_general = VkImageMemoryBarrier(
                    sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                    dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    newLayout=VK_IMAGE_LAYOUT_GENERAL,
                    image=curr_image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                )
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0,
                    None,
                    0,
                    None,
                    2,
                    [prev_to_general, curr_back_to_general],
                )

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_layout, 0, 1, [compute_set], 0, None)
            pc_mode = ffi.new("int[]", [0 if args.mode == "raw" else 1])
            vkCmdPushConstants(cmd, compute_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, pc_mode)
            vkCmdDispatch(cmd, (width + 15) // 16, (height + 15) // 16, 1)

            out_to_sample = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                image=out_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [out_to_sample],
            )
            prev_to_transfer = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                image=prev_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            curr_to_src = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image=curr_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                2,
                [prev_to_transfer, curr_to_src],
            )
            vkCmdCopyImage(
                cmd,
                curr_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                prev_image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [VkImageCopy(
                    srcSubresource=VkImageSubresourceLayers(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    srcOffset=VkOffset3D(0, 0, 0),
                    dstSubresource=VkImageSubresourceLayers(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    dstOffset=VkOffset3D(0, 0, 0),
                    extent=VkExtent3D(width=width, height=height, depth=1),
                )],
            )
            prev_back_to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=prev_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            curr_back_to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=curr_image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                2,
                [prev_back_to_general, curr_back_to_general],
            )

        clear = VkClearValue(color=VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0]))
        vkCmdBeginRenderPass(
            cmd,
            VkRenderPassBeginInfo(
                sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=render_pass,
                framebuffer=framebuffers[image_index],
                renderArea=VkRect2D(VkOffset2D(0, 0), VkExtent2D(swap_width, swap_height)),
                clearValueCount=1,
                pClearValues=[clear],
            ),
            VK_SUBPASS_CONTENTS_INLINE,
        )
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_layout, 0, 1, [graphics_set], 0, None)
        vkCmdDraw(cmd, 3, 1, 0, 0)
        vkCmdEndRenderPass(cmd)

        vkEndCommandBuffer(cmd)
        record_elapsed_ms = (time.perf_counter() - record_start) * 1000.0
        cpu_record_ms += max(0.0, record_elapsed_ms - copy_ms)

        submit_start = time.perf_counter()
        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=[image_available[frame_idx]],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[cmd],
            signalSemaphoreCount=1,
            pSignalSemaphores=[render_done[frame_idx]],
        )
        vkQueueSubmit(queue, 1, [submit], in_flight[frame_idx])

        present = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[render_done[frame_idx]],
            swapchainCount=1,
            pSwapchains=[swapchain],
            pImageIndices=[image_index],
        )
        queue_present(queue, present)
        gpu_submit_ms += (time.perf_counter() - submit_start) * 1000.0

        frame += 1
        first_frame = False
        cpu_total_ms += (time.perf_counter() - frame_start) * 1000.0
        if frame % args.stats_every == 0:
            fps = frame / max(1e-6, time.time() - start)
            avg_prep = cpu_prep_ms / frame
            avg_wait = cpu_wait_ms / frame
            avg_copy = cpu_copy_ms / frame
            avg_record = cpu_record_ms / frame
            avg_submit = gpu_submit_ms / frame
            avg_total = cpu_total_ms / frame
            print(
                f"frame {frame} fps {fps:.2f} "
                f"cpu_prep_ms {avg_prep:.2f} "
                f"wait_ms {avg_wait:.2f} copy_ms {avg_copy:.2f} record_ms {avg_record:.2f} "
                f"submit_ms {avg_submit:.2f} total_ms {avg_total:.2f}"
            )

    vkDeviceWaitIdle(device)
    if frame:
        avg_prep = cpu_prep_ms / frame
        avg_wait = cpu_wait_ms / frame
        avg_copy = cpu_copy_ms / frame
        avg_record = cpu_record_ms / frame
        avg_submit = gpu_submit_ms / frame
        avg_total = cpu_total_ms / frame
        print(
            f"done frames {frame} avg_cpu_prep_ms {avg_prep:.2f} "
            f"avg_wait_ms {avg_wait:.2f} avg_copy_ms {avg_copy:.2f} "
            f"avg_record_ms {avg_record:.2f} avg_submit_ms {avg_submit:.2f} "
            f"avg_total_ms {avg_total:.2f}"
        )

    for mem in staging_mems:
        vkUnmapMemory(device, mem)
    for sem in render_done:
        vkDestroySemaphore(device, sem, None)
    for sem in image_available:
        vkDestroySemaphore(device, sem, None)
    for fence in in_flight:
        vkDestroyFence(device, fence, None)
    for fb in framebuffers:
        vkDestroyFramebuffer(device, fb, None)
    vkDestroyPipeline(device, compute_pipeline, None)
    vkDestroyPipeline(device, nv12_pipeline, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, compute_layout, None)
    vkDestroyPipelineLayout(device, nv12_layout, None)
    vkDestroyPipelineLayout(device, graphics_layout, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    vkDestroyDescriptorSetLayout(device, compute_set_layout, None)
    vkDestroyDescriptorSetLayout(device, nv12_set_layout, None)
    vkDestroyDescriptorSetLayout(device, graphics_set_layout, None)
    vkDestroyRenderPass(device, render_pass, None)
    vkDestroyShaderModule(device, diff_module, None)
    vkDestroyShaderModule(device, nv12_module, None)
    vkDestroyShaderModule(device, frag_module, None)
    vkDestroyShaderModule(device, vert_module, None)
    vkDestroySampler(device, sampler, None)
    vkDestroyImageView(device, curr_view, None)
    vkDestroyImageView(device, prev_view, None)
    vkDestroyImageView(device, out_view, None)
    vkDestroyImage(device, curr_image, None)
    vkDestroyImage(device, prev_image, None)
    vkDestroyImage(device, out_image, None)
    vkFreeMemory(device, curr_memory, None)
    vkFreeMemory(device, prev_memory, None)
    vkFreeMemory(device, out_memory, None)
    for buf in staging_buffers:
        vkDestroyBuffer(device, buf, None)
    for mem in staging_mems:
        vkFreeMemory(device, mem, None)
    if dmabuf_buffer is not None:
        vkDestroyBuffer(device, dmabuf_buffer, None)
    if dmabuf_memory is not None:
        vkFreeMemory(device, dmabuf_memory, None)
    for view in swap_views:
        vkDestroyImageView(device, view, None)
    destroy_swapchain(device, swapchain, None)
    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDevice(device, None)
    destroy_surface(instance, surface, None)
    vkDestroyInstance(instance, None)
    glfw.destroy_window(window)
    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
