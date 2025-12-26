from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from vulkan import *  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"python-vulkan is required: {exc}") from exc

try:
    from .dmabuf_export import export_dmabuf
except Exception:
    from dmabuf_export import export_dmabuf


def _fourcc(a: str, b: str, c: str, d: str) -> int:
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


def _fourcc_str(code: int) -> str:
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


DRM_FORMAT_NV12 = _fourcc("N", "V", "1", "2")
DRM_FORMAT_P010 = _fourcc("P", "0", "1", "0")
DRM_FORMAT_R16 = _fourcc("R", "1", "6", " ")
DRM_FORMAT_MOD_INVALID = 0xFFFFFFFFFFFFFFFF

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _ensure_spirv(spv_path: Path, src_path: Path) -> None:
    if spv_path.exists():
        try:
            if spv_path.stat().st_mtime >= src_path.stat().st_mtime:
                return
        except OSError:
            pass
    subprocess.run(["glslc", str(src_path), "-o", str(spv_path)], check=True)


def _read_spirv_bytes(spv_path: Path) -> bytes:
    return spv_path.read_bytes()


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


def _import_dmabuf_buffers(info: dict, fds: List[int]) -> None:
    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vaapi_dmabuf_stub",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 0, 0),
    )
    instance = vkCreateInstance(
        VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        ),
        None,
    )
    devices = vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise RuntimeError("no Vulkan physical devices found")
    physical_device = devices[0]

    required_exts = [
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_EXT_external_memory_dma_buf",
    ]
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
            enabledExtensionCount=len(required_exts),
            ppEnabledExtensionNames=required_exts,
        ),
        None,
    )
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    imported = []
    for obj_idx in range(info["nb_objects"]):
        size = int(info["objects"][obj_idx]["size"])
        fd = fds[obj_idx]
        buf = vkCreateBuffer(
            device,
            VkBufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size,
                usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        mem_req = vkGetBufferMemoryRequirements(device, buf)
        memory_type = _find_memory_type(physical_device, mem_req.memoryTypeBits)
        import_info = VkImportMemoryFdInfoKHR(
            sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            fd=fd,
        )
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_req.size,
            memoryTypeIndex=memory_type,
            pNext=import_info,
        )
        memory = vkAllocateMemory(device, alloc_info, None)
        vkBindBufferMemory(device, buf, memory, 0)
        imported.append((buf, memory, size))

    print("dmabuf buffer-import OK")
    for idx, (_, __, size) in enumerate(imported):
        print(f"  object {idx}: size={size} bytes")

    if info["nb_objects"] < 1 or info["nb_planes"] < 2:
        raise RuntimeError("buffer import requires at least one object and two planes")

    shader_src = SHADER_DIR / "nv12_to_rgba.comp"
    shader_spv = shader_src.with_suffix(".spv")
    _ensure_spirv(shader_spv, shader_src)
    shader_bytes = _read_spirv_bytes(shader_spv)
    shader_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_bytes),
            pCode=shader_bytes,
        ),
        None,
    )

    set_layout = vkCreateDescriptorSetLayout(
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
    push_size = 7 * 4
    pipeline_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[
                VkPushConstantRange(
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=push_size,
                )
            ],
        ),
        None,
    )
    pipeline = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        [
            VkComputePipelineCreateInfo(
                sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=VkPipelineShaderStageCreateInfo(
                    sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=VK_SHADER_STAGE_COMPUTE_BIT,
                    module=shader_module,
                    pName=ffi.new("char[]", b"main"),
                ),
                layout=pipeline_layout,
            )
        ],
        None,
    )[0]

    out_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        extent=VkExtent3D(width=info["width"], height=info["height"], depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    out_image = vkCreateImage(device, out_info, None)
    out_reqs = vkGetImageMemoryRequirements(device, out_image)
    out_type = _find_memory_type(physical_device, out_reqs.memoryTypeBits)
    out_mem = vkAllocateMemory(
        device,
        VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=out_reqs.size,
            memoryTypeIndex=out_type,
        ),
        None,
    )
    vkBindImageMemory(device, out_image, out_mem, 0)
    out_view = vkCreateImageView(
        device,
        VkImageViewCreateInfo(
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
        ),
        None,
    )

    descriptor_pool = vkCreateDescriptorPool(
        device,
        VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=2,
            pPoolSizes=[
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1),
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=1),
            ],
            maxSets=1,
        ),
        None,
    )
    descriptor_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[set_layout],
        ),
    )[0]
    vkUpdateDescriptorSets(
        device,
        2,
        [
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[VkDescriptorBufferInfo(buffer=imported[0][0], offset=0, range=imported[0][2])],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[VkDescriptorImageInfo(imageView=out_view, imageLayout=VK_IMAGE_LAYOUT_GENERAL)],
            ),
        ],
        0,
        None,
    )

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
        [
            VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
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
        ],
    )
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        [descriptor_set],
        0,
        None,
    )
    if info["drm_format"] == DRM_FORMAT_NV12:
        fmt_id = 0
    else:
        fmt_id = 1
    pc_values = ffi.new(
        "uint32_t[]",
        [
            info["width"],
            info["height"],
            info["planes"][0]["offset"],
            info["planes"][1]["offset"],
            info["planes"][0]["pitch"],
            info["planes"][1]["pitch"],
            fmt_id,
        ],
    )
    vkCmdPushConstants(
        cmd,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        push_size,
        pc_values,
    )
    vkCmdDispatch(cmd, (info["width"] + 15) // 16, (info["height"] + 15) // 16, 1)
    vkEndCommandBuffer(cmd)
    vkQueueSubmit(
        queue,
        1,
        [
            VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd],
            )
        ],
        VK_NULL_HANDLE,
    )
    vkQueueWaitIdle(queue)
    print("nv12/p010 buffer -> rgba conversion OK")

    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    vkDestroyImageView(device, out_view, None)
    vkDestroyImage(device, out_image, None)
    vkFreeMemory(device, out_mem, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, pipeline_layout, None)
    vkDestroyDescriptorSetLayout(device, set_layout, None)
    vkDestroyShaderModule(device, shader_module, None)

    for buf, memory, _ in imported:
        vkDestroyBuffer(device, buf, None)
        vkFreeMemory(device, memory, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal VAAPI dmabuf -> Vulkan import stub.")
    parser.add_argument("video", type=Path, help="Path to input video.")
    parser.add_argument("--vaapi-device", default="/dev/dri/renderD128")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Max seconds to wait for export.")
    parser.add_argument(
        "--force-linear",
        action="store_true",
        help="Force VAAPI download/upload to get a linear dmabuf when modifiers are implicit.",
    )
    parser.add_argument("--debug", action="store_true", help="Print exporter debug logs.")
    parser.add_argument(
        "--drm-device",
        default=None,
        help="DRM device path for force-linear uploads (defaults to --vaapi-device).",
    )
    args = parser.parse_args()

    if args.force_linear:
        if not args.drm_device or not os.path.exists(args.drm_device):
            raise SystemExit(
                f"--force-linear requested but --drm-device '{args.drm_device}' does not exist.\n"
                "Your /dev/dri only exposes a render node; disable --force-linear or expose /dev/dri/card*.\n"
                "Tip: `ls -la /dev/dri`"
            )

    info, fds, _log = export_dmabuf(
        args.video,
        args.vaapi_device,
        force_linear=args.force_linear,
        drm_device=args.drm_device,
        timeout_s=args.timeout_s,
        debug=args.debug,
    )

    try:
        _import_dmabuf_image(info, fds)
    except RuntimeError as exc:
        print(f"image import failed: {exc}")
        print("falling back to dmabuf buffer import")
        _import_dmabuf_buffers(info, fds)
    print(
        f"imported dmabuf frame {info['width']}x{info['height']} "
        f"format=0x{info['drm_format']:08x} modifier=0x{info['objects'][0]['modifier']:x}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
