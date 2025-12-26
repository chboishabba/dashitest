from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from vulkan import *
from vulkan import ffi


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


def _find_queue_family_index(physical_device: VkPhysicalDevice) -> int:
    props = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for idx, prop in enumerate(props):
        if prop.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return idx
    raise RuntimeError("No compute-capable queue family found.")


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


def _dump_ppm(path: Path, rgba: np.ndarray, width: int, height: int) -> None:
    rgb = rgba.reshape(height, width, 4)[:, :, :3]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(rgb.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Vulkan compute storage image sample.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dump", type=Path, default=None, help="Write output PPM to this path.")
    args = parser.parse_args()

    shader_path = Path(__file__).parent / "shaders" / "write_image.spv"
    if not shader_path.exists():
        print(f"Missing shader: {shader_path}. Compile write_image.comp with glslc first.", file=sys.stderr)
        return 1

    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vulkan_compute_image",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 0, 0),
    )
    instance_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )
    instance = vkCreateInstance(instance_info, None)

    physical_device = _select_physical_device(instance)
    queue_family_index = _find_queue_family_index(physical_device)

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
    )
    device = vkCreateDevice(physical_device, device_info, None)
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    width = args.width
    height = args.height
    image_info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        format=VK_FORMAT_R8G8B8A8_UNORM,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        samples=VK_SAMPLE_COUNT_1_BIT,
        tiling=VK_IMAGE_TILING_OPTIMAL,
        usage=VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    )
    image = vkCreateImage(device, image_info, None)
    mem_reqs = vkGetImageMemoryRequirements(device, image)
    mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    mem_type = _find_memory_type(
        mem_props,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    alloc_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_reqs.size,
        memoryTypeIndex=mem_type,
    )
    image_memory = vkAllocateMemory(device, alloc_info, None)
    vkBindImageMemory(device, image, image_memory, 0)

    view_info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=image,
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
    image_view = vkCreateImageView(device, view_info, None)

    shader_code = _read_spirv(shader_path)
    shader_info = VkShaderModuleCreateInfo(
        sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(shader_code),
        pCode=shader_code,
    )
    shader_module = vkCreateShaderModule(device, shader_info, None)

    descriptor_set_layout_binding = VkDescriptorSetLayoutBinding(
        binding=0,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        descriptorCount=1,
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    )
    descriptor_set_layout_info = VkDescriptorSetLayoutCreateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=1,
        pBindings=[descriptor_set_layout_binding],
    )
    descriptor_set_layout = vkCreateDescriptorSetLayout(device, descriptor_set_layout_info, None)

    push_constant_range = VkPushConstantRange(
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        offset=0,
        size=4,
    )
    pipeline_layout_info = VkPipelineLayoutCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1,
        pSetLayouts=[descriptor_set_layout],
        pushConstantRangeCount=1,
        pPushConstantRanges=[push_constant_range],
    )
    pipeline_layout = vkCreatePipelineLayout(device, pipeline_layout_info, None)

    stage_info = VkPipelineShaderStageCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName="main",
    )
    pipeline_info = VkComputePipelineCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=stage_info,
        layout=pipeline_layout,
    )
    pipeline = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

    pool_size = VkDescriptorPoolSize(
        type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        descriptorCount=1,
    )
    descriptor_pool_info = VkDescriptorPoolCreateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        poolSizeCount=1,
        pPoolSizes=[pool_size],
        maxSets=1,
    )
    descriptor_pool = vkCreateDescriptorPool(device, descriptor_pool_info, None)
    descriptor_set_alloc = VkDescriptorSetAllocateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=descriptor_pool,
        descriptorSetCount=1,
        pSetLayouts=[descriptor_set_layout],
    )
    descriptor_set = vkAllocateDescriptorSets(device, descriptor_set_alloc)[0]

    image_info_desc = VkDescriptorImageInfo(
        imageView=image_view,
        imageLayout=VK_IMAGE_LAYOUT_GENERAL,
    )
    write = VkWriteDescriptorSet(
        sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=descriptor_set,
        dstBinding=0,
        descriptorCount=1,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        pImageInfo=[image_info_desc],
    )
    vkUpdateDescriptorSets(device, 1, [write], 0, None)

    staging_size = width * height * 4
    staging_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=staging_size,
        usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

    command_pool_info = VkCommandPoolCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
    )
    command_pool = vkCreateCommandPool(device, command_pool_info, None)
    command_buffer_alloc = VkCommandBufferAllocateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=command_pool,
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    command_buffer = vkAllocateCommandBuffers(device, command_buffer_alloc)[0]

    begin_info = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vkBeginCommandBuffer(command_buffer, begin_info)

    to_general = VkImageMemoryBarrier(
        sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        srcAccessMask=0,
        dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
        oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        newLayout=VK_IMAGE_LAYOUT_GENERAL,
        image=image,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    vkCmdPipelineBarrier(
        command_buffer,
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

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        [descriptor_set],
        0,
        None,
    )
    pc_data = ffi.new("uint32_t[]", [args.frame])
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc_data,
    )
    group_x = (width + 15) // 16
    group_y = (height + 15) // 16
    vkCmdDispatch(command_buffer, group_x, group_y, 1)

    to_transfer = VkImageMemoryBarrier(
        sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
        dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
        oldLayout=VK_IMAGE_LAYOUT_GENERAL,
        newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        image=image,
        subresourceRange=VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    vkCmdPipelineBarrier(
        command_buffer,
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

    region = VkBufferImageCopy(
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
        imageExtent=VkExtent3D(width, height, 1),
    )
    vkCmdCopyImageToBuffer(
        command_buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        staging_buffer,
        1,
        [region],
    )

    vkEndCommandBuffer(command_buffer)

    submit_info = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[command_buffer],
    )
    vkQueueSubmit(queue, 1, [submit_info], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)

    mapped = vkMapMemory(device, staging_memory, 0, staging_size, 0)
    mapped_buf = np.frombuffer(mapped, dtype=np.uint8, count=staging_size)
    rgba = mapped_buf.copy()
    vkUnmapMemory(device, staging_memory)

    if args.dump:
        _dump_ppm(args.dump, rgba, width, height)
        print(f"Wrote {args.dump}")
    else:
        print("Readback complete; use --dump to write a PPM file.")

    vkDestroyBuffer(device, staging_buffer, None)
    vkFreeMemory(device, staging_memory, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, pipeline_layout, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    vkDestroyShaderModule(device, shader_module, None)
    vkDestroyImageView(device, image_view, None)
    vkDestroyImage(device, image, None)
    vkFreeMemory(device, image_memory, None)
    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
