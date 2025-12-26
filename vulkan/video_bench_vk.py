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


SHADER_DIR = Path(__file__).resolve().parents[1] / "vulkan_compute" / "shaders"


def ffprobe_video(path: Path) -> Tuple[int, int, int]:
    """Return (width, height, nb_frames|0)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "0").isdigit() else 0
    return width, height, nb_frames


def decode_gray(path: Path, width: int, height: int, max_frames: int) -> np.ndarray:
    """Decode video to grayscale raw frames via ffmpeg; returns array [T,H,W] uint8."""
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-vframes",
        str(max_frames),
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    frame_size = width * height
    total_pixels = len(raw)
    if total_pixels % frame_size != 0:
        raise ValueError("Decoded byte count not divisible by frame size; ffmpeg decode mismatch.")
    frames = total_pixels // frame_size
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(frames, height, width)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Vulkan video bench preview (GPU path).")
    parser.add_argument("video", type=Path, help="Path to input video (e.g., MP4).")
    parser.add_argument("--frames", type=int, default=240, help="Max frames to decode for the preview.")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    width, height, nb_frames = ffprobe_video(args.video)
    max_frames = args.frames
    if nb_frames:
        max_frames = min(max_frames, nb_frames)
    frames = decode_gray(args.video, width, height, max_frames)
    if frames.size == 0:
        raise SystemExit("No frames decoded.")

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
    image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    texture_image = vkCreateImage(device, image_info, None)
    img_reqs = vkGetImageMemoryRequirements(device, texture_image)
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
    texture_memory = vkAllocateMemory(device, img_alloc, None)
    vkBindImageMemory(device, texture_image, texture_memory, 0)
    texture_view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=texture_image,
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
    texture_view = vkCreateImageView(device, texture_view_info, None)

    sampler_info = VkSamplerCreateInfo(
        sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        magFilter=VK_FILTER_NEAREST,
        minFilter=VK_FILTER_NEAREST,
        addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    )
    sampler = vkCreateSampler(device, sampler_info, None)

    vert_shader = SHADER_DIR / "preview.vert.spv"
    frag_shader = SHADER_DIR / "preview.frag.spv"
    for spv in (vert_shader, frag_shader):
        if not spv.exists():
            raise SystemExit(f"Missing shader: {spv}. Compile with glslc.")

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
            poolSizeCount=1,
            pPoolSizes=[
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1)
            ],
            maxSets=1,
        ),
        None,
    )
    graphics_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[graphics_set_layout],
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
                        imageView=texture_view,
                        imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )
                ],
            )
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

    staging_size = width * height * 4
    staging_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=staging_size,
        usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    staging_buffer = vkCreateBuffer(device, staging_info, None)
    staging_reqs = vkGetBufferMemoryRequirements(device, staging_buffer)
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
    staging_memory = vkAllocateMemory(device, staging_alloc, None)
    vkBindBufferMemory(device, staging_buffer, staging_memory, 0)
    mapped = vkMapMemory(device, staging_memory, 0, staging_size, 0)
    mapped_buf = np.frombuffer(mapped, dtype=np.uint8, count=staging_size)

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

    image_available = vkCreateSemaphore(device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None)
    render_done = vkCreateSemaphore(device, VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO), None)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    first_frame = True
    frame = 0
    start = time.time()
    while not glfw.window_should_close(window) and frame < frames.shape[0]:
        glfw.poll_events()
        image_index = acquire_next_image(
            device, swapchain, 0xFFFFFFFFFFFFFFFF, image_available, VK_NULL_HANDLE
        )
        cmd = command_buffers[image_index]
        vkResetCommandBuffer(cmd, 0)
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))

        gray = frames[frame]
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        mapped_buf[:] = rgba.ravel()

        to_transfer = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=VK_IMAGE_LAYOUT_UNDEFINED if first_frame else VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            image=texture_image,
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
            [to_transfer],
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
            staging_buffer,
            texture_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            [copy_region],
        )

        to_sample = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
            oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image=texture_image,
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
        vkQueueWaitIdle(queue)

        frame += 1
        first_frame = False
        if frame % 60 == 0:
            fps = frame / max(1e-6, time.time() - start)
            print(f"frame {frame} fps {fps:.2f}")

    vkDeviceWaitIdle(device)

    vkUnmapMemory(device, staging_memory)
    vkDestroySemaphore(device, render_done, None)
    vkDestroySemaphore(device, image_available, None)
    for fb in framebuffers:
        vkDestroyFramebuffer(device, fb, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, graphics_layout, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    vkDestroyDescriptorSetLayout(device, graphics_set_layout, None)
    vkDestroyRenderPass(device, render_pass, None)
    vkDestroyShaderModule(device, frag_module, None)
    vkDestroyShaderModule(device, vert_module, None)
    vkDestroySampler(device, sampler, None)
    vkDestroyImageView(device, texture_view, None)
    vkDestroyImage(device, texture_image, None)
    vkFreeMemory(device, texture_memory, None)
    vkDestroyBuffer(device, staging_buffer, None)
    vkFreeMemory(device, staging_memory, None)
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
