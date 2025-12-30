from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from vulkan import *  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"python-vulkan is required: {exc}") from exc

from vulkan import ffi

try:
    from .decode_backend import ffprobe_video
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from decode_backend import ffprobe_video

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _ensure_spirv(spv_path: Path, src_path: Path) -> None:
    if not src_path.exists():
        raise SystemExit(f"Missing shader source: {src_path}")
    if spv_path.exists():
        try:
            if spv_path.stat().st_mtime >= src_path.stat().st_mtime:
                return
        except OSError:
            pass
    subprocess.run(["glslc", str(src_path), "-o", str(spv_path)], check=True)


def _read_spirv_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _choose_queue_family(physical_device) -> int:
    families = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for idx, fam in enumerate(families):
        if fam.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return idx
    for idx, fam in enumerate(families):
        if fam.queueFlags & VK_QUEUE_GRAPHICS_BIT:
            return idx
    return 0


def _find_memory_type(mem_props, type_bits: int, required_flags: int) -> int:
    for idx in range(mem_props.memoryTypeCount):
        if type_bits & (1 << idx):
            flags = mem_props.memoryTypes[idx].propertyFlags
            if (flags & required_flags) == required_flags:
                return idx
    raise RuntimeError("No compatible memory type found.")


def _check_zero(buf, size: int, full: bool) -> bool:
    view = memoryview(ffi.buffer(buf, size))
    if not full:
        sample = view[: min(4096, size)]
        return all(b == 0 for b in sample)
    return all(b == 0 for b in view)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU symbol stream zero-writer stub.")
    parser.add_argument("video", type=Path, help="Path to input video.")
    parser.add_argument("--block", type=int, default=16, help="Block size.")
    parser.add_argument("--planes", type=int, default=4, help="Ternary planes per channel.")
    parser.add_argument("--channels", type=int, default=1, help="Channel count (Y=1, YCoCg=3).")
    parser.add_argument("--check", action="store_true", help="Verify all output bytes are zero.")
    args = parser.parse_args()

    width, height, _ = ffprobe_video(args.video)
    blocks_x = (width + args.block - 1) // args.block
    blocks_y = (height + args.block - 1) // args.block
    n_blocks = blocks_x * blocks_y
    block_words = n_blocks * 4  # 4 uints per block
    plane_count = args.planes * args.channels
    trit_count = plane_count * width * height

    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="symbol_stream_stub",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_MAKE_VERSION(1, 0, 0),
    )
    instance = vkCreateInstance(VkInstanceCreateInfo(sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info), None)
    physical_device = vkEnumeratePhysicalDevices(instance)[0]
    mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    queue_family_index = _choose_queue_family(physical_device)
    device = vkCreateDevice(
        physical_device,
        VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[
                VkDeviceQueueCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=queue_family_index,
                    queueCount=1,
                    pQueuePriorities=[1.0],
                )
            ],
        ),
        None,
    )
    queue = vkGetDeviceQueue(device, queue_family_index, 0)

    block_size_bytes = block_words * 4
    trit_size_bytes = trit_count * 4

    block_buf = vkCreateBuffer(
        device,
        VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=block_size_bytes,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        ),
        None,
    )
    trit_buf = vkCreateBuffer(
        device,
        VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=trit_size_bytes,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        ),
        None,
    )
    block_reqs = vkGetBufferMemoryRequirements(device, block_buf)
    trit_reqs = vkGetBufferMemoryRequirements(device, trit_buf)
    host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    block_type = _find_memory_type(mem_props, block_reqs.memoryTypeBits, host_flags)
    trit_type = _find_memory_type(mem_props, trit_reqs.memoryTypeBits, host_flags)
    block_mem = vkAllocateMemory(
        device,
        VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=block_reqs.size,
            memoryTypeIndex=block_type,
        ),
        None,
    )
    trit_mem = vkAllocateMemory(
        device,
        VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=trit_reqs.size,
            memoryTypeIndex=trit_type,
        ),
        None,
    )
    vkBindBufferMemory(device, block_buf, block_mem, 0)
    vkBindBufferMemory(device, trit_buf, trit_mem, 0)

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
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ),
            ],
        ),
        None,
    )
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
                    size=8,
                )
            ],
        ),
        None,
    )
    shader_src = SHADER_DIR / "symbols_zero.comp"
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

    pool = vkCreateDescriptorPool(
        device,
        VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=2,
            pPoolSizes=[
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1),
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1),
            ],
            maxSets=1,
        ),
        None,
    )
    descriptor_set = vkAllocateDescriptorSets(
        device,
        VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=pool,
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
                pBufferInfo=[VkDescriptorBufferInfo(buffer=block_buf, offset=0, range=block_size_bytes)],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[VkDescriptorBufferInfo(buffer=trit_buf, offset=0, range=trit_size_bytes)],
            ),
        ],
        0,
        None,
    )

    cmd_pool = vkCreateCommandPool(
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
            commandPool=cmd_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ),
    )[0]
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
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
    pc = ffi.new("uint32_t[]", [block_words, trit_count])
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc)
    total = max(block_words, trit_count)
    groups = (total + 255) // 256
    vkCmdDispatch(cmd, groups, 1, 1)
    vkEndCommandBuffer(cmd)
    vkQueueSubmit(queue, 1, [VkSubmitInfo(sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[cmd])], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)

    block_map = vkMapMemory(device, block_mem, 0, block_size_bytes, 0)
    trit_map = vkMapMemory(device, trit_mem, 0, trit_size_bytes, 0)
    block_zero = _check_zero(block_map, block_size_bytes, args.check)
    trit_zero = _check_zero(trit_map, trit_size_bytes, args.check)
    vkUnmapMemory(device, block_mem)
    vkUnmapMemory(device, trit_mem)

    print(f"blocks={n_blocks} block_words={block_words} block_bytes={block_size_bytes}")
    print(f"planes={plane_count} trits={trit_count} trit_bytes={trit_size_bytes}")
    print(f"zero_writer_ok blocks={block_zero} trits={trit_zero}")

    vkDestroyCommandPool(device, cmd_pool, None)
    vkDestroyDescriptorPool(device, pool, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, pipeline_layout, None)
    vkDestroyDescriptorSetLayout(device, set_layout, None)
    vkDestroyShaderModule(device, shader_module, None)
    vkDestroyBuffer(device, block_buf, None)
    vkDestroyBuffer(device, trit_buf, None)
    vkFreeMemory(device, block_mem, None)
    vkFreeMemory(device, trit_mem, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
