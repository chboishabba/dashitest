from __future__ import annotations

import ctypes
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
        hint = (
            f"VK_ICD_FILENAMES={icd_env}" if icd_env else "VK_ICD_FILENAMES not set"
        )
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


def _mapped_buffer(mapped: object, size: int) -> np.ndarray:
    try:
        return np.frombuffer(mapped, dtype=np.uint8, count=size)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported mapped buffer type: {type(mapped)}") from exc


def main() -> int:
    shader_path = Path(__file__).parent / "shaders" / "add.spv"
    if not shader_path.exists():
        print(f"Missing shader: {shader_path}. Compile add.comp with glslc first.", file=sys.stderr)
        return 1

    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vulkan_compute_buffer",
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

    element_count = 1024
    data = np.arange(element_count, dtype=np.int32)
    buffer_size = data.nbytes

    buffer_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=buffer_size,
        usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    buffer = vkCreateBuffer(device, buffer_info, None)
    mem_reqs = vkGetBufferMemoryRequirements(device, buffer)
    mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    mem_type = _find_memory_type(
        mem_props,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    alloc_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_reqs.size,
        memoryTypeIndex=mem_type,
    )
    memory = vkAllocateMemory(device, alloc_info, None)
    vkBindBufferMemory(device, buffer, memory, 0)

    mapped = vkMapMemory(device, memory, 0, buffer_size, 0)
    mapped_buf = _mapped_buffer(mapped, buffer_size)
    mapped_buf[:] = data.view(np.uint8)
    vkUnmapMemory(device, memory)

    shader_code = _read_spirv(shader_path)
    shader_info = VkShaderModuleCreateInfo(
        sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(shader_code),
        pCode=shader_code,
    )
    shader_module = vkCreateShaderModule(device, shader_info, None)

    descriptor_set_layout_binding = VkDescriptorSetLayoutBinding(
        binding=0,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
        size=ctypes.sizeof(ctypes.c_uint32),
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
        type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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

    buffer_info = VkDescriptorBufferInfo(
        buffer=buffer,
        offset=0,
        range=buffer_size,
    )
    write = VkWriteDescriptorSet(
        sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=descriptor_set,
        dstBinding=0,
        descriptorCount=1,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        pBufferInfo=[buffer_info],
    )
    vkUpdateDescriptorSets(device, 1, [write], 0, None)

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
    pc_data = ffi.new("uint32_t[]", [element_count])
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        pc_data,
    )
    group_count = (element_count + 255) // 256
    vkCmdDispatch(command_buffer, group_count, 1, 1)
    vkEndCommandBuffer(command_buffer)

    submit_info = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[command_buffer],
    )
    vkQueueSubmit(queue, 1, [submit_info], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)

    mapped = vkMapMemory(device, memory, 0, buffer_size, 0)
    result = np.empty_like(data)
    mapped_buf = _mapped_buffer(mapped, buffer_size)
    result.view(np.uint8)[:] = mapped_buf
    vkUnmapMemory(device, memory)

    print("first 10 results:", result[:10])

    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, pipeline_layout, None)
    vkDestroyDescriptorPool(device, descriptor_pool, None)
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    vkDestroyShaderModule(device, shader_module, None)
    vkDestroyBuffer(device, buffer, None)
    vkFreeMemory(device, memory, None)
    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
