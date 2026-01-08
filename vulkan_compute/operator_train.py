#!/usr/bin/env python3
"""
Run the OperatorLearner training loop fully on Vulkan compute and export diagnostics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import struct
import sys
from pathlib import Path

import numpy as np
from vulkan import *
from vulkan import ffi

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compute_buffer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train OperatorLearner on Vulkan.")
    ap.add_argument(
        "--energy-seq",
        type=Path,
        required=True,
        help="Input `E_seq.npy` band-energy sequence.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of training steps (uses seq[:-1]).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="SGD learning rate applied in shader.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Contractivity threshold (spectral proxy) applied per row.",
    )
    ap.add_argument(
        "--sheet-energy",
        type=Path,
        default=Path("dashilearn/sheet_energy.npy"),
        help="Path to write the Vulkan sheet energy trace ([steps,B]).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/operator_metrics.json"),
        help="Base path for metrics JSON (timestamp appended).",
    )
    ap.add_argument(
        "--spv",
        type=Path,
        default=Path(__file__).resolve().parent / "shaders/operator_step.spv",
        help="Compiled operator_step SPIR-V module.",
    )
    return ap.parse_args()


def _create_buffer(
    device: VkDevice,
    physical_device: VkPhysicalDevice,
    size: int,
    usage: int,
    properties: int = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
) -> tuple[VkBuffer, VkDeviceMemory]:
    buffer_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    buffer = vkCreateBuffer(device, buffer_info, None)
    reqs = vkGetBufferMemoryRequirements(device, buffer)
    mem_type = compute_buffer._find_memory_type(
        vkGetPhysicalDeviceMemoryProperties(physical_device),
        reqs.memoryTypeBits,
        properties,
    )
    alloc_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=reqs.size,
        memoryTypeIndex=mem_type,
    )
    memory = vkAllocateMemory(device, alloc_info, None)
    vkBindBufferMemory(device, buffer, memory, 0)
    return buffer, memory


def _write_array(device: VkDevice, memory: VkDeviceMemory, array: np.ndarray) -> None:
    size = array.nbytes
    mapped = vkMapMemory(device, memory, 0, size, 0)
    ffi.memmove(mapped, array.tobytes(), size)
    vkUnmapMemory(device, memory)


def _map_array(
    device: VkDevice,
    memory: VkDeviceMemory,
    size: int,
    dtype: np.dtype,
) -> np.ndarray:
    mapped = vkMapMemory(device, memory, 0, size, 0)
    view = np.frombuffer(ffi.buffer(mapped, size), dtype=dtype, count=size // np.dtype(dtype).itemsize)
    return view


def _timestamped_path(path: Path) -> Path:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if path.suffix:
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")
    return path.with_name(f"{path.name}_{ts}")


def main() -> None:
    args = parse_args()
    seq = np.load(args.energy_seq).astype(np.float32)
    if seq.ndim != 2:
        raise ValueError("E_seq must be shape [T,B]")
    T, B = seq.shape
    if B > 256:
        raise ValueError("band count exceeds shader MAX_BANDS (256)")
    step_count = min(args.steps, T - 1)
    if step_count <= 0:
        raise ValueError("E_seq must contain at least two steps")

    shader_path = args.spv
    if not shader_path.exists():
        raise FileNotFoundError(f"{shader_path} not found; compile the shader with glslangValidator.")

    required_extensions = [b"VK_KHR_portability_enumeration"]
    instance_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        flags=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        pApplicationInfo=VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="operator_train_vulkan",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="none",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 0),
        ),
        enabledExtensionCount=len(required_extensions),
        ppEnabledExtensionNames=required_extensions,
    )
    instance = vkCreateInstance(instance_info, None)
    physical_device = compute_buffer._select_physical_device(instance)
    queue_family = compute_buffer._find_queue_family_index(physical_device)
    queue_info = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family,
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    device = vkCreateDevice(
        physical_device,
        VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledLayerCount=0,
        ),
        None,
    )
    queue = vkGetDeviceQueue(device, queue_family, 0)

    seq_buffer_size = seq.nbytes
    w_size = B * B * 4
    b_size = B * 4
    metrics_size = 4 * 4
    sheet_size = step_count * B * 4

    seq_buffer, seq_memory = _create_buffer(
        device,
        physical_device,
        seq_buffer_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    )
    _write_array(device, seq_memory, seq)

    weights_buffer, weights_memory = _create_buffer(
        device,
        physical_device,
        w_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    )
    weights_init = np.random.uniform(-0.01, 0.01, size=(B, B)).astype(np.float32).ravel()
    _write_array(device, weights_memory, weights_init)

    bias_buffer, bias_memory = _create_buffer(
        device,
        physical_device,
        b_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    )
    bias_init = np.zeros(B, dtype=np.float32)
    _write_array(device, bias_memory, bias_init)

    metrics_buffer, metrics_memory = _create_buffer(
        device,
        physical_device,
        metrics_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    )
    _write_array(device, metrics_memory, np.zeros(4, dtype=np.float32))

    sheet_buffer, sheet_memory = _create_buffer(
        device,
        physical_device,
        sheet_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    )

    spirv_bytes = shader_path.read_bytes()
    shader_module = vkCreateShaderModule(
        device,
        VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spirv_bytes),
            pCode=spirv_bytes,
        ),
        None,
    )

    bindings = []
    for idx in range(5):
        bindings.append(
            VkDescriptorSetLayoutBinding(
                binding=idx,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
        )
    descriptor_layout = vkCreateDescriptorSetLayout(
        device,
        VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        ),
        None,
    )

    push_range = VkPushConstantRange(
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        offset=0,
        size=6 * 4,
    )
    pipeline_layout = vkCreatePipelineLayout(
        device,
        VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
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
                    layout=pipeline_layout,
                    stage=VkPipelineShaderStageCreateInfo(
                        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        stage=VK_SHADER_STAGE_COMPUTE_BIT,
                        module=shader_module,
                        pName=b"main",
                    ),
                )
            ],
            None,
        )[0]

    pool = vkCreateDescriptorPool(
        device,
        VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=[
                VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)
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
            pSetLayouts=[descriptor_layout],
        ),
    )[0]

    buffer_infos = [
        VkDescriptorBufferInfo(buffer=seq_buffer, offset=0, range=seq_buffer_size),
        VkDescriptorBufferInfo(buffer=weights_buffer, offset=0, range=w_size),
        VkDescriptorBufferInfo(buffer=bias_buffer, offset=0, range=b_size),
        VkDescriptorBufferInfo(buffer=metrics_buffer, offset=0, range=metrics_size),
        VkDescriptorBufferInfo(buffer=sheet_buffer, offset=0, range=sheet_size),
    ]
    writes = []
    for idx, info in enumerate(buffer_infos):
        writes.append(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=idx,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[info],
            )
        )
    vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    command_pool = vkCreateCommandPool(
        device,
        VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queue_family,
        ),
        None,
    )
    command_buffer = vkAllocateCommandBuffers(
        device,
        VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ),
    )[0]

    metrics_view = _map_array(device, metrics_memory, metrics_size, np.float32)

    def record_and_submit(step: int) -> tuple[float, float, float]:
        vkResetCommandBuffer(command_buffer, 0)
        vkBeginCommandBuffer(
            command_buffer,
            VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO),
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
        packed = struct.pack(
            "4i2f",
            step,
            B,
            B,
            B,
            args.lr,
            args.alpha,
        )
        vkCmdPushConstants(
            command_buffer,
            pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            len(packed),
            packed,
        )
        vkCmdDispatch(command_buffer, 1, 1, 1)
        vkEndCommandBuffer(command_buffer)
        fence = vkCreateFence(device, VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        vkQueueSubmit(queue, 1, [submit_info], fence)
        vkWaitForFences(device, 1, [fence], VK_TRUE, 1_000_000_000)
        vkDestroyFence(device, fence, None)
        vkInvalidateMappedMemoryRanges(
            device,
            1,
            [
                VkMappedMemoryRange(
                    sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                    memory=metrics_memory,
                    offset=0,
                    size=metrics_size,
                )
            ],
        )
        metrics = np.copy(metrics_view[:4])
        return metrics[1], metrics[2], metrics[3]

    print(f"Running {step_count} steps (seq length {T}) on Vulkan")
    losses = []
    max_norms = []
    avg_norms = []
    for step in range(step_count):
        loss, max_norm, avg_norm = record_and_submit(step)
        losses.append(float(loss))
        max_norms.append(float(max_norm))
        avg_norms.append(float(avg_norm))
        print(f" step {step+1}/{step_count} loss={loss:.6e} max_norm={max_norm:.6e} avg_norm={avg_norm:.6e}")

    vkQueueWaitIdle(queue)

    # Snapshot sheet energy
    vkUnmapMemory(device, metrics_memory)
    sheet_data = np.frombuffer(
        ffi.buffer(vkMapMemory(device, sheet_memory, 0, sheet_size, 0), sheet_size),
        dtype=np.float32,
        count=step_count * B,
    ).copy()
    vkUnmapMemory(device, sheet_memory)
    sheet_data = sheet_data.reshape(step_count, B)
    args.sheet_energy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.sheet_energy, sheet_data)
    print(f"Wrote sheet energy trace to {args.sheet_energy}")

    metrics_json = {
        "operator_baseline_vulkan_steps": step_count,
        "operator_baseline_vulkan_loss": float(losses[-1]) if losses else 0.0,
        "operator_baseline_vulkan_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "operator_baseline_vulkan_max_norm": float(max_norms[-1]) if max_norms else 0.0,
        "operator_baseline_vulkan_avg_norm": float(np.mean(avg_norms)) if avg_norms else 0.0,
        "operator_baseline_vulkan_lr": args.lr,
        "operator_baseline_vulkan_alpha": args.alpha,
        "operator_baseline_vulkan_train_seq_len": T,
        "operator_baseline_vulkan_bands": B,
    }
    out_path = _timestamped_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(metrics_json, fh, indent=2, sort_keys=True)
    print(f"Saved metrics to {out_path}")

    vkDestroyCommandPool(device, command_pool, None)
    vkDestroyDescriptorPool(device, pool, None)
    vkDestroyPipeline(device, pipeline, None)
    vkDestroyPipelineLayout(device, pipeline_layout, None)
    vkDestroyDescriptorSetLayout(device, descriptor_layout, None)
    vkDestroyShaderModule(device, shader_module, None)
    for buffer, memory in [
        (seq_buffer, seq_memory),
        (weights_buffer, weights_memory),
        (bias_buffer, bias_memory),
        (metrics_buffer, metrics_memory),
        (sheet_buffer, sheet_memory),
    ]:
        vkDestroyBuffer(device, buffer, None)
        vkFreeMemory(device, memory, None)
    vkDestroyDevice(device, None)
    vkDestroyInstance(instance, None)


if __name__ == "__main__":
    main()
