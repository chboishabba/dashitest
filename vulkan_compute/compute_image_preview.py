from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import struct
import subprocess
import shutil
import datetime
import re
from pathlib import Path

import numpy as np

try:
    import glfw  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("glfw is required for preview: pip install glfw") from exc

from vulkan import *
from vulkan import ffi
import vulkan as vkmod


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
        if not (prop.queueFlags & VK_QUEUE_COMPUTE_BIT):
            continue
        if surface_support is not None:
            if surface_support(physical_device, idx, surface):
                return idx
        else:
            if glfw.get_physical_device_presentation_support(instance, physical_device, idx):
                return idx
    raise RuntimeError("No queue family supports graphics+compute+present.")


def _find_compute_queue_family_index(physical_device: VkPhysicalDevice) -> int:
    props = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for idx, prop in enumerate(props):
        if prop.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return idx
    raise RuntimeError("No queue family supports compute.")


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


_TS_RE = re.compile(r".*_[0-9]{8}T[0-9]{6}Z$")


def _timestamped_path(path: Path) -> Path:
    if _TS_RE.match(path.stem):
        return path
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if path.suffix:
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")
    return path.with_name(f"{path.name}_{ts}")


def _mapped_buffer(mapped: object, size: int, dtype: np.dtype) -> np.ndarray:
    """
    Create a NumPy view over mapped Vulkan memory without rebinding an ffi buffer.
    """
    try:
        buf = memoryview(mapped)
    except TypeError:
        try:
            buf = ffi.buffer(mapped, size)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Unsupported mapped buffer type: {type(mapped)}") from exc
    count = size // np.dtype(dtype).itemsize
    return np.frombuffer(buf, dtype=dtype, count=count)


def _stretch_sheet_values(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    src_h, src_w = arr.shape
    if src_h == 0 or src_w == 0:
        raise ValueError("Sheet data has zero width or height.")
    if (src_h, src_w) == (target_h, target_w):
        return arr
    rows = math.ceil(target_h / src_h)
    cols = math.ceil(target_w / src_w)
    stretched = np.repeat(np.repeat(arr, rows, axis=0), cols, axis=1)
    return stretched[:target_h, :target_w]


def _choose_surface_format(formats: list[VkSurfaceFormatKHR]) -> VkSurfaceFormatKHR:
    for fmt in formats:
        if fmt.format == VK_FORMAT_B8G8R8A8_UNORM:
            return fmt
    return formats[0]


def _choose_present_mode(modes: list[int]) -> int:
    if VK_PRESENT_MODE_MAILBOX_KHR in modes:
        return VK_PRESENT_MODE_MAILBOX_KHR
    return VK_PRESENT_MODE_FIFO_KHR


def main() -> int:
    parser = argparse.ArgumentParser(description="Vulkan compute image preview (GLFW).")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--sheet", action="store_true", help="Use sheet expand+fade compute shader.")
    parser.add_argument("--sheet-w", type=int, default=32, help="Sheet width (cells).")
    parser.add_argument("--sheet-h", type=int, default=32, help="Sheet height (cells).")
    parser.add_argument("--block-px", type=int, default=16, help="Pixels per sheet cell.")
    parser.add_argument("--alpha", type=float, default=0.97, help="Fade factor for accumulator.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Visual range min.")
    parser.add_argument("--vmax", type=float, default=1.0, help="Visual range max.")
    parser.add_argument("--no-clamp", action="store_true", help="Disable vmin/vmax clamping.")
    parser.add_argument("--record-video", action="store_true", help="Record frames to a video file.")
    parser.add_argument("--record-out", default="sheet_preview.mp4", help="Output path for recording.")
    parser.add_argument("--record-fps", type=int, default=60, help="Sampling FPS for recording.")
    parser.add_argument("--sheet-data", type=str, default=None, help="Optional .npy file exported by the learner for live sheet data.")
    parser.add_argument("--sheet-data-interval", type=float, default=0.5, help="Seconds between reloading the sheet data file.")
    parser.add_argument("--headless", action="store_true", help="Run compute-only without a window.")
    parser.add_argument("--smoke", action="store_true", help="Create instance/device and exit early.")
    args = parser.parse_args()

    sheet_mode = args.sheet
    record_enabled = args.record_video
    if record_enabled and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH (required for --record-video).")

    sheet_data_path = Path(args.sheet_data) if args.sheet_data else None
    sheet_reload_interval = max(0.0, args.sheet_data_interval)
    sheet_data_values = None
    sheet_last_reload = 0.0
    sheet_data_mtime = 0.0
    sheet_w = args.sheet_w
    sheet_h = args.sheet_h
    if sheet_mode and sheet_data_path is not None and sheet_data_path.exists():
        try:
            sheet_arr = np.load(sheet_data_path)
            if sheet_arr.ndim == 2:
                sheet_h, sheet_w = sheet_arr.shape
                sheet_data_values = sheet_arr.astype(np.float32)
                sheet_data_mtime = sheet_data_path.stat().st_mtime
            else:
                print(f"warning: {sheet_data_path} is not 2D; ignoring", file=sys.stderr)
        except Exception as exc:
            print(f"warning: failed to load sheet data from {sheet_data_path}: {exc}", file=sys.stderr)
    window = None
    surface = None
    if not args.headless:
        if not glfw.init():
            raise SystemExit("Failed to initialize GLFW.")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(args.width, args.height, "Vulkan Compute Preview", None, None)
        if not window:
            glfw.terminate()
            raise SystemExit("Failed to create GLFW window.")

    required_exts = glfw.get_required_instance_extensions() if window else []
    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vulkan_compute_preview",
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

    destroy_surface = None
    physical_device = _select_physical_device(instance)
    if window:
        surface_ptr = ffi.new("VkSurfaceKHR*")
        result = glfw.create_window_surface(instance, window, None, surface_ptr)
        if result != VK_SUCCESS:
            raise RuntimeError(f"glfwCreateWindowSurface failed with VkResult={result}.")
        surface = surface_ptr[0]
        destroy_surface = _load_instance_func(instance, "vkDestroySurfaceKHR")
        queue_family_index = _find_queue_family_index(instance, physical_device, surface)
    else:
        queue_family_index = _find_compute_queue_family_index(physical_device)

    queue_priority = 1.0
    queue_info = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=[queue_priority],
    )
    enabled_exts = [VK_KHR_SWAPCHAIN_EXTENSION_NAME] if window else []
    device_info = VkDeviceCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_info],
        enabledExtensionCount=len(enabled_exts),
        ppEnabledExtensionNames=enabled_exts,
    )
    device = vkCreateDevice(physical_device, device_info, None)
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    if args.smoke:
        vkDestroyDevice(device, None)
        if destroy_surface:
            destroy_surface(instance, surface, None)
        vkDestroyInstance(instance, None)
        if window:
            glfw.destroy_window(window)
            glfw.terminate()
        return 0

    surface_format = None
    present_mode = None
    swapchain = None
    swap_images = []
    swap_views = []
    acquire_next_image = None
    queue_present = None
    destroy_swapchain = None
    width = args.width
    height = args.height
    if window:
        destroy_surface = _load_instance_func(instance, "vkDestroySurfaceKHR")
        surface_caps_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        surface_formats_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
        present_modes_fn = _load_instance_func(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR")

        surface_caps = surface_caps_fn(physical_device, surface)
        formats = list(surface_formats_fn(physical_device, surface))
        present_modes = list(present_modes_fn(physical_device, surface))
        surface_format = _choose_surface_format(formats)
        present_mode = _choose_present_mode(present_modes)
        extent = surface_caps.currentExtent
        width = extent.width if extent.width != 0xFFFFFFFF else args.width
        height = extent.height if extent.height != 0xFFFFFFFF else args.height

        image_count = max(surface_caps.minImageCount + 1, 2)
        if surface_caps.maxImageCount > 0:
            image_count = min(image_count, surface_caps.maxImageCount)

        swapchain_info = VkSwapchainCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=surface,
            minImageCount=image_count,
            imageFormat=surface_format.format,
            imageColorSpace=surface_format.colorSpace,
            imageExtent=VkExtent2D(width, height),
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
    image_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    if record_enabled:
        image_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=image_usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    compute_image = vkCreateImage(device, image_info, None)
    img_reqs = vkGetImageMemoryRequirements(device, compute_image)
    img_type = _find_memory_type(
        mem_props,
        img_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    img_alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=img_reqs.size,
        memoryTypeIndex=img_type,
    )
    compute_memory = vkAllocateMemory(device, img_alloc, None)
    vkBindImageMemory(device, compute_image, compute_memory, 0)
    compute_view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=compute_image,
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
    compute_view = vkCreateImageView(device, compute_view_info, None)
    accum_image = None
    accum_memory = None
    accum_view = None
    sheet_buffer = None
    sheet_memory = None
    sheet_mapped = None
    sheet_data = None
    sheet_size = 0
    if sheet_mode:
        accum_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=VK_IMAGE_TYPE_2D,
            format=VK_FORMAT_R32_SFLOAT,
            extent=VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )
        accum_image = vkCreateImage(device, accum_info, None)
        accum_reqs = vkGetImageMemoryRequirements(device, accum_image)
        accum_type = _find_memory_type(
            mem_props,
            accum_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        accum_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=accum_reqs.size,
            memoryTypeIndex=accum_type,
        )
        accum_memory = vkAllocateMemory(device, accum_alloc, None)
        vkBindImageMemory(device, accum_image, accum_memory, 0)
        accum_view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=accum_image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=VK_FORMAT_R32_SFLOAT,
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        accum_view = vkCreateImageView(device, accum_view_info, None)

    if sheet_mode:
        sheet_size = sheet_w * sheet_h * 4
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=sheet_size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        sheet_buffer = vkCreateBuffer(device, buffer_info, None)
        sheet_reqs = vkGetBufferMemoryRequirements(device, sheet_buffer)
        sheet_type = _find_memory_type(
            mem_props,
            sheet_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        sheet_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=sheet_reqs.size,
            memoryTypeIndex=sheet_type,
        )
        sheet_memory = vkAllocateMemory(device, sheet_alloc, None)
        vkBindBufferMemory(device, sheet_buffer, sheet_memory, 0)
        sheet_mapped = vkMapMemory(device, sheet_memory, 0, sheet_size, 0)
        sheet_data = _mapped_buffer(sheet_mapped, sheet_size, np.float32)
        if sheet_data_values is None:
            sheet_base = (
                np.arange(sheet_h, dtype=np.float32)[:, None]
                + np.arange(sheet_w, dtype=np.float32)[None, :]
            )
        else:
            sheet_base = None
    else:
        sheet_base = None

    record_buffer = None
    record_memory = None
    record_mapped = None
    record_view = None
    record_proc = None
    record_interval = None
    record_next_time = None
    if record_enabled:
        record_size = width * height * 4
        record_out = _timestamped_path(Path(args.record_out))
        print(f"recording -> {record_out}")
        record_buffer = vkCreateBuffer(
            device,
            VkBufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=record_size,
                usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        record_reqs = vkGetBufferMemoryRequirements(device, record_buffer)
        record_type = _find_memory_type(
            mem_props,
            record_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        record_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=record_reqs.size,
            memoryTypeIndex=record_type,
        )
        record_memory = vkAllocateMemory(device, record_alloc, None)
        vkBindBufferMemory(device, record_buffer, record_memory, 0)
        record_mapped = vkMapMemory(device, record_memory, 0, record_size, 0)
        record_view = _mapped_buffer(record_mapped, record_size, np.uint8)
        record_interval = 1.0 / max(1, args.record_fps)
        record_next_time = time.monotonic()
        record_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            str(args.record_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(record_out),
        ]
        record_proc = subprocess.Popen(record_cmd, stdin=subprocess.PIPE)

    if sheet_mode and (args.width != args.sheet_w * args.block_px or args.height != args.sheet_h * args.block_px):
        print(
            "warning: output size != sheet_w*block_px; sheet will not fill the image.",
            file=sys.stderr,
        )
    compute_shader = Path(__file__).parent / "shaders" / (
        "sheet_expand_fade.spv" if sheet_mode else "write_image.spv"
    )
    shaders = [compute_shader]
    if window:
        shaders.extend(
            [
                Path(__file__).parent / "shaders" / "preview.vert.spv",
                Path(__file__).parent / "shaders" / "preview.frag.spv",
            ]
        )
    for spv in shaders:
        if not spv.exists():
            raise SystemExit(f"Missing shader: {spv}. Compile with glslc.")

    compute_spirv = _read_spirv(compute_shader)
    compute_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(compute_spirv),
            pCode=compute_spirv,
        ),
        None,
    )
    vert_module = None
    frag_module = None
    if window:
        vert_shader = shaders[1]
        frag_shader = shaders[2]
        vert_module = vkCreateShaderModule(
            device,
            VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(_read_spirv(vert_shader)),
                pCode=_read_spirv(vert_shader),
            ),
            None,
        )
        frag_module = vkCreateShaderModule(
            device,
            VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(_read_spirv(frag_shader)),
                pCode=_read_spirv(frag_shader),
            ),
            None,
        )

    if sheet_mode:
        compute_bindings = [
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
            VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
    else:
        compute_bindings = [
            VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
        ]
    compute_set_layout = vkCreateDescriptorSetLayout(
        device,
        VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(compute_bindings),
            pBindings=compute_bindings,
        ),
        None,
    )
    graphics_set_layout = None
    if window:
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

    compute_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[compute_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[
                VkPushConstantRange(
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=36 if sheet_mode else 4,
                )
            ],
        ),
        None,
    )
    graphics_layout = None
    if window:
        graphics_layout = vkCreatePipelineLayout(
            device,
            VkPipelineLayoutCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=ffi.new("VkDescriptorSetLayout[]", [graphics_set_layout]),
            ),
            None,
        )

    compute_stage_name = ffi.new("char[]", b"main")
    compute_stage = VkPipelineShaderStageCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=compute_module,
        pName=compute_stage_name,
    )
    compute_pipeline_info = VkComputePipelineCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=compute_stage,
        layout=compute_layout,
    )
    compute_pipeline_infos = ffi.new("VkComputePipelineCreateInfo[]", [compute_pipeline_info])
    compute_pipeline = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        compute_pipeline_infos,
        None,
    )[0]

    render_pass = None
    pipeline = None
    if window:
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
        viewports = ffi.new("VkViewport[]", [VkViewport(0, 0, width, height, 0.0, 1.0)])
        scissors = ffi.new("VkRect2D[]", [VkRect2D(VkOffset2D(0, 0), VkExtent2D(width, height))])
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

    pool_sizes = []
    if sheet_mode:
        pool_sizes.append(VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1))
        pool_sizes.append(VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=2))
    else:
        pool_sizes.append(VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=1))
    max_sets = 1
    if window:
        pool_sizes.append(
            VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1)
        )
        max_sets = 2
    descriptor_pool = vkCreateDescriptorPool(
        device,
        VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
            maxSets=max_sets,
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
    graphics_set = None
    sampler = None
    if window:
        sampler_info = VkSamplerCreateInfo(
            sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter=VK_FILTER_NEAREST,
            minFilter=VK_FILTER_NEAREST,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        )
        sampler = vkCreateSampler(device, sampler_info, None)
        graphics_set = vkAllocateDescriptorSets(
            device,
            VkDescriptorSetAllocateInfo(
                sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[graphics_set_layout],
            ),
        )[0]

    writes = []
    if sheet_mode:
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    VkDescriptorBufferInfo(
                        buffer=sheet_buffer,
                        offset=0,
                        range=sheet_size,
                    )
                ],
            )
        )
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[
                    VkDescriptorImageInfo(imageView=accum_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)
                ],
            )
        )
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=2,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[
                    VkDescriptorImageInfo(imageView=compute_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)
                ],
            )
        )
    else:
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=compute_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[VkDescriptorImageInfo(imageView=compute_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
            )
        )
    if window:
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=graphics_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[
                    VkDescriptorImageInfo(
                        sampler=sampler,
                        imageView=compute_view,
                        imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )
                ],
            )
        )
    vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    framebuffers = []
    if window:
        for view in swap_views:
            framebuffers.append(
                vkCreateFramebuffer(
                    device,
                    VkFramebufferCreateInfo(
                        sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                        renderPass=render_pass,
                        attachmentCount=1,
                        pAttachments=[view],
                        width=width,
                        height=height,
                        layers=1,
                    ),
                    None,
                )
            )

    command_pool = vkCreateCommandPool(
        device,
        VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
        ),
        None,
    )
    command_buffer_count = len(swap_images) if window else 1
    command_buffers = vkAllocateCommandBuffers(
        device,
        VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=command_buffer_count,
        ),
    )

    image_available = None
    render_done = None
    if window:
        image_available = vkCreateSemaphore(
            device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None
        )
        render_done = vkCreateSemaphore(
            device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None
        )

    frame = 0
    start = time.time()
    while (not window or not glfw.window_should_close(window)) and frame < args.frames:
        now = time.monotonic()
        capture_frame = False
        if record_enabled and record_interval is not None:
            if now >= record_next_time:
                capture_frame = True
                while record_next_time <= now:
                    record_next_time += record_interval
        if sheet_mode and sheet_data_path is not None and sheet_reload_interval >= 0.0:
            if now >= sheet_last_reload + sheet_reload_interval:
                sheet_last_reload = now
                try:
                    if not sheet_data_path.exists():
                        continue
                    stat = sheet_data_path.stat()
                    mtime = stat.st_mtime
                    if mtime > sheet_data_mtime:
                        arr = np.load(sheet_data_path)
                        if arr.ndim == 2:
                            if arr.shape != (sheet_h, sheet_w):
                                try:
                                    arr = _stretch_sheet_values(arr, (sheet_h, sheet_w))
                                except ValueError as exc:
                                    print(
                                        f"warning: cannot stretch sheet data from {arr.shape}: {exc}",
                                        file=sys.stderr,
                                    )
                                    continue
                            sheet_data_values = arr.astype(np.float32)
                            sheet_data_mtime = mtime
                        else:
                            print(
                                f"warning: {sheet_data_path} is not 2D; ignoring",
                                file=sys.stderr,
                            )
                    #
                except Exception as exc:
                    print(f"warning: failed to reload sheet data: {exc}", file=sys.stderr)
        if sheet_mode:
            phase = frame * 0.15
            if sheet_data_values is not None:
                sheet_data[:] = sheet_data_values.ravel()
            else:
                values = 0.5 + 0.5 * np.sin(phase + sheet_base * 0.2)
                sheet_data[:] = values.ravel()
        if window:
            glfw.poll_events()
            image_index = acquire_next_image(
                device, swapchain, 0xFFFFFFFFFFFFFFFF, image_available, VK_NULL_HANDLE
            )
            cmd = command_buffers[image_index]
        else:
            cmd = command_buffers[0]
        vkResetCommandBuffer(cmd, 0)
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))

        if frame == 0:
            to_general = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=compute_image,
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
                [to_general],
            )
            if sheet_mode:
                to_general_accum = VkImageMemoryBarrier(
                    sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                    oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=VK_IMAGE_LAYOUT_GENERAL,
                    image=accum_image,
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
                    [to_general_accum],
                )
                clear = VkClearColorValue(float32=[0.0, 0.0, 0.0, 0.0])
                vkCmdClearColorImage(
                    cmd,
                    accum_image,
                    VK_IMAGE_LAYOUT_GENERAL,
                    clear,
                    1,
                    [VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    )],
                )

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_layout, 0, 1, [compute_set], 0, None)
        if sheet_mode:
            pc_data = struct.pack(
                "<5I3fI",
                sheet_w,
                sheet_h,
                width,
                height,
                args.block_px,
                float(args.alpha),
                float(args.vmin),
                float(args.vmax),
                0 if args.no_clamp else 1,
            )
            pc_ptr = ffi.new("char[]", pc_data)
            vkCmdPushConstants(cmd, compute_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, len(pc_data), pc_ptr)
        else:
            pc_data = ffi.new("uint32_t[]", [frame])
            vkCmdPushConstants(cmd, compute_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, pc_data)
        vkCmdDispatch(cmd, (width + 15) // 16, (height + 15) // 16, 1)
        if capture_frame:
            to_transfer = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=compute_image,
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
                1,
                [to_transfer],
            )
            vkCmdCopyImageToBuffer(
                cmd,
                compute_image,
                VK_IMAGE_LAYOUT_GENERAL,
                record_buffer,
                1,
                [
                    VkBufferImageCopy(
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
                ],
            )
        if window:
            to_sample = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT if capture_frame else VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_GENERAL,
                newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                image=compute_image,
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
                [to_sample],
            )

            clear = VkClearValue(color=VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0]))
            vkCmdBeginRenderPass(
                cmd,
                VkRenderPassBeginInfo(
                    sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    renderPass=render_pass,
                    framebuffer=framebuffers[image_index],
                    renderArea=VkRect2D(VkOffset2D(0, 0), VkExtent2D(width, height)),
                    clearValueCount=1,
                    pClearValues=[clear],
                ),
                VK_SUBPASS_CONTENTS_INLINE,
            )
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_layout, 0, 1, [graphics_set], 0, None
            )
            vkCmdDraw(cmd, 3, 1, 0, 0)
            vkCmdEndRenderPass(cmd)

            to_general_again = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                newLayout=VK_IMAGE_LAYOUT_GENERAL,
                image=compute_image,
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
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [to_general_again],
            )

        vkEndCommandBuffer(cmd)

        if window:
            submit = VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                waitSemaphoreCount=1,
                pWaitSemaphores=[image_available],
                pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                commandBufferCount=1,
                pCommandBuffers=[cmd],
                signalSemaphoreCount=1,
                pSignalSemaphores=[render_done],
            )
            vkQueueSubmit(queue, 1, [submit], VK_NULL_HANDLE)

            present = VkPresentInfoKHR(
                sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                waitSemaphoreCount=1,
                pWaitSemaphores=[render_done],
                swapchainCount=1,
                pSwapchains=[swapchain],
                pImageIndices=[image_index],
            )
            queue_present(queue, present)
        else:
            submit = VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd],
            )
            vkQueueSubmit(queue, 1, [submit], VK_NULL_HANDLE)
        vkQueueWaitIdle(queue)
        if capture_frame and record_proc and record_proc.stdin:
            try:
                record_proc.stdin.write(record_view)
            except BrokenPipeError:
                record_enabled = False

        frame += 1
        if frame % 60 == 0:
            fps = frame / max(1e-6, time.time() - start)
            print(f"frame {frame} fps {fps:.2f}")

    if record_proc:
        if record_proc.stdin:
            record_proc.stdin.close()
        record_proc.wait()

    vkDeviceWaitIdle(device)

    if render_done:
        vkDestroySemaphore(device, render_done, None)
    if image_available:
        vkDestroySemaphore(device, image_available, None)
    for fb in framebuffers:
        vkDestroyFramebuffer(device, fb, None)
    if pipeline:
        vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipeline(device, compute_pipeline, None)
    if graphics_layout:
        vkDestroyPipelineLayout(device, graphics_layout, None)
    vkDestroyPipelineLayout(device, compute_layout, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    if graphics_set_layout:
        vkDestroyDescriptorSetLayout(device, graphics_set_layout, None)
    vkDestroyDescriptorSetLayout(device, compute_set_layout, None)
    if render_pass:
        vkDestroyRenderPass(device, render_pass, None)
    if frag_module:
        vkDestroyShaderModule(device, frag_module, None)
    if vert_module:
        vkDestroyShaderModule(device, vert_module, None)
    vkDestroyShaderModule(device, compute_module, None)
    if sampler:
        vkDestroySampler(device, sampler, None)
    if sheet_mode:
        if sheet_mapped is not None:
            vkUnmapMemory(device, sheet_memory)
        if accum_view:
            vkDestroyImageView(device, accum_view, None)
        if accum_image:
            vkDestroyImage(device, accum_image, None)
        if accum_memory:
            vkFreeMemory(device, accum_memory, None)
        if sheet_buffer:
            vkDestroyBuffer(device, sheet_buffer, None)
        if sheet_memory:
            vkFreeMemory(device, sheet_memory, None)
    if record_buffer is not None:
        if record_mapped is not None:
            vkUnmapMemory(device, record_memory)
        if record_buffer:
            vkDestroyBuffer(device, record_buffer, None)
        if record_memory:
            vkFreeMemory(device, record_memory, None)
    vkDestroyImageView(device, compute_view, None)
    vkDestroyImage(device, compute_image, None)
    vkFreeMemory(device, compute_memory, None)
    for view in swap_views:
        vkDestroyImageView(device, view, None)
    if destroy_swapchain:
        destroy_swapchain(device, swapchain, None)
    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDevice(device, None)
    if destroy_surface:
        destroy_surface(instance, surface, None)
    vkDestroyInstance(instance, None)
    if window:
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
