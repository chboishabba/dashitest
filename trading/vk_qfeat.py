from __future__ import annotations

import os
import pathlib
import ctypes
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import math

try:
    from trading.features.quotient import compute_qfeat
except ModuleNotFoundError:
    from features.quotient import compute_qfeat

ELL_BETA_VOL = 0.15
ELL_BETA_DD = 0.15
ELL_BETA_BURST = 0.10
ELL_BETA_ACORR = 0.30
ELL_BETA_CURV = 0.10

SCHEMA_VERSION = 1


class WeightsSSBO(ctypes.Structure):
    _fields_ = [
        ("cvar_alpha", ctypes.c_float),
        ("hazard_threshold", ctypes.c_float),
        ("tau_on", ctypes.c_float),
        ("tau_off", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("hazard_veto", ctypes.c_uint),
        ("_pad0", ctypes.c_uint),
        ("_pad1", ctypes.c_uint),
        ("score_weights", ctypes.c_float * 8),
        ("opt_tenor_weights", ctypes.c_float * 5),
        ("opt_mny_weights", ctypes.c_float * 5),
        ("schema_version", ctypes.c_uint),
    ]


def _default_weights_ssbo(*, eps: float) -> WeightsSSBO:
    w = WeightsSSBO()
    w.cvar_alpha = 0.10
    w.hazard_threshold = 2.0
    w.tau_on = 0.5
    w.tau_off = 0.49
    w.epsilon = float(eps)
    w.hazard_veto = 1
    w._pad0 = 0
    w._pad1 = 0
    for i in range(8):
        w.score_weights[i] = 0.0
    for i in range(5):
        w.opt_tenor_weights[i] = 0.0
        w.opt_mny_weights[i] = 0.0
    w.schema_version = SCHEMA_VERSION
    return w


@dataclass
class Params:
    num_series: int
    T: int
    w1: int = 64
    w2: int = 256
    price_stride: int = 0
    volume_stride: int = 0
    flags: int = 0
    eps: float = 1e-6
    nan_squash: float = 0.0
    burst_clip: float = 0.0
    dt: float = 0.0

    def __post_init__(self):
        if self.price_stride == 0:
            self.price_stride = self.T
        if self.volume_stride == 0:
            self.volume_stride = self.T

    def meta0(self) -> np.ndarray:
        return np.array([self.num_series, self.T, self.w1, self.w2], dtype=np.uint32)

    def meta1(self) -> np.ndarray:
        return np.array([self.price_stride, self.volume_stride, self.flags, 0], dtype=np.uint32)

    def fmeta0(self) -> np.ndarray:
        return np.array([self.eps, self.nan_squash, self.burst_clip, 0.0], dtype=np.float32)

    def fmeta1(self) -> np.ndarray:
        return np.array([self.dt, 0.0, 0.0, 0.0], dtype=np.float32)


class QFeatTape:
    """
    Memory-mapped qfeat tape with ABI `[vol_ratio, curvature, drawdown,
    burstiness, acorr_1, var_ratio, ℓ, reserved]`.
    """

    def __init__(self, path: str, num_series: int, T: int):
        self.path = str(path)
        self.num_series = int(num_series)
        self.T = int(T)
        self.shape = (self.num_series, self.T, 8)
        self.mm = np.memmap(self.path, dtype=np.float32, mode="r", shape=self.shape)

    @classmethod
    def from_existing(cls, path: str, rows: Optional[int] = None) -> "QFeatTape":
        mm = np.memmap(path, dtype=np.float32, mode="r")
        if rows is None:
            if mm.ndim != 3 or mm.shape[2] != 8:
                raise ValueError("tape must be float32 with last dim 8")
            num_series, T, _ = mm.shape
            return cls(path, num_series, T)
        if rows <= 0:
            raise ValueError("rows must be positive")
        if mm.size % (rows * 8) != 0:
            raise ValueError("tape size does not align with rows")
        num_series = mm.size // (rows * 8)
        return cls(path, num_series, rows)

    def qfeat(self, series: int, t: int) -> np.ndarray:
        return np.array(self.mm[series, t, :6], dtype=np.float32, copy=True)

    def ell(self, series: int, t: int) -> float:
        return float(self.mm[series, t, 6])


def _compute_cpu_tape(
    prices: np.ndarray,
    *,
    w1: int,
    w2: int,
    eps: float,
    nan_squash: float,
    out_memmap: np.memmap,
) -> None:
    S, T = prices.shape
    for s in range(S):
        series = prices[s]
        for t in range(T):
            if t < w2:
                continue
            window = series[(t - w2) : (t + 1)]
            q = compute_qfeat(window, w1=w1, w2=w2, eps=eps)
            out_memmap[s, t, :6] = q
            ell = _ell_from_qfeat(
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(q[3]),
                float(q[4]),
                nan_squash,
            )
            out_memmap[s, t, 6] = ell
            out_memmap[s, t, 7] = 0.0


def _ell_from_qfeat(
    vol_ratio: float,
    curvature: float,
    drawdown: float,
    burstiness: float,
    acorr_1: float,
    nan_squash: float,
) -> float:
    vol_pen = math.log1p(vol_ratio)
    dd_pen = math.log1p(drawdown)
    burst_pen = math.log1p(abs(burstiness - 1.0))
    acorr_pen = 1.0 - abs(acorr_1)
    penalty = (
        ELL_BETA_VOL * vol_pen
        + ELL_BETA_DD * dd_pen
        + ELL_BETA_BURST * burst_pen
        + ELL_BETA_ACORR * acorr_pen
        + ELL_BETA_CURV * curvature
    )
    ell = math.exp(-penalty)
    if not math.isfinite(ell):
        return float(nan_squash)
    if ell < 0.0:
        return 0.0
    if ell > 1.0:
        return 1.0
    return float(ell)


def build_feature_tape(
    *,
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    out_path: str,
    w1: int = 64,
    w2: int = 256,
    eps: float = 1e-6,
    nan_squash: float = 0.0,
    flags: int = 0,
    force: bool = False,
    backend: str = "cpu",
    shader_path: str = "vulkan_shaders/qfeat.comp",
    spv_path: str = "vulkan_shaders/qfeat.spv",
    vk_icd: Optional[str] = None,
    fp64_returns: bool = True,
    timing_debug: bool = False,
) -> QFeatTape:
    """
    Build a memmap tape (S, T, 8) of qfeat records. The current
    implementation supports CPU and Vulkan backends. Use backend="vulkan"
    to dispatch the compute shader described in TRADER_CONTEXT2.md.
    """
    prices = np.asarray(prices, dtype=np.float32, order="C")
    if prices.ndim != 2:
        raise ValueError("prices must be 2D (series, timesteps)")
    S, T = prices.shape
    if volumes is not None:
        volumes_arr = np.asarray(volumes, dtype=np.float32, order="C")
        if volumes_arr.shape != prices.shape:
            raise ValueError("volume shape must match price shape")

    out_path = pathlib.Path(out_path)
    if out_path.exists():
        if force:
            out_path.unlink()
        else:
            raise FileExistsError(f"{out_path} already exists (use force=True to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(S, T, 8))
    mm[:] = 0.0
    mm.flush()

    timing_info: dict[str, float] | None = None
    if backend == "cpu":
        _compute_cpu_tape(
            prices,
            w1=w1,
            w2=w2,
            eps=eps,
            nan_squash=nan_squash,
            out_memmap=mm,
        )
        mm.flush()
    elif backend == "vulkan":
        shader_root = pathlib.Path(__file__).resolve().parent
        shader_file = pathlib.Path(shader_path)
        spv_file = pathlib.Path(spv_path)
        if not shader_file.is_absolute():
            shader_file = shader_root / shader_file
        if not spv_file.is_absolute():
            spv_file = shader_root / spv_file
        timing_info = _run_vulkan_tape(
            prices=prices,
            volumes=volumes,
            out_path=out_path,
            w1=w1,
            w2=w2,
            eps=eps,
            nan_squash=nan_squash,
            flags=flags,
            shader_path=shader_file,
            spv_path=spv_file,
            vk_icd=vk_icd,
            fp64_returns=fp64_returns,
            timing_debug=timing_debug,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")

    tape = QFeatTape(str(out_path), S, T)
    if timing_debug and timing_info:
        host_setup = timing_info["host_setup"]
        gpu_compute = timing_info["gpu_compute"]
        host_teardown = timing_info["host_teardown"]
        print(
            f"[vk_qfeat] timing host_setup={host_setup:.3f}s gpu={gpu_compute:.3f}s "
            f"host_teardown={host_teardown:.3f}s"
        )
    return tape


def _compile_shader(
    shader_path: pathlib.Path,
    spv_path: pathlib.Path,
    *,
    defines: Optional[list[str]] = None,
) -> None:
    if not shader_path.exists():
        raise FileNotFoundError(shader_path)
    if spv_path.exists() and spv_path.stat().st_mtime >= shader_path.stat().st_mtime:
        return
    define_args = []
    for define in defines or []:
        define_args.append(f"-D{define}")
    cmd = ["glslc", *define_args, str(shader_path), "-o", str(spv_path)]
    result = os.spawnvp(os.P_WAIT, "glslc", cmd)
    if result != 0:
        raise RuntimeError(f"glslc failed with exit code {result}")


def _create_buffer(device, physical_device, size: int):
    from vulkan import (
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        VkBufferCreateInfo,
        VkMemoryAllocateInfo,
        vkAllocateMemory,
        vkBindBufferMemory,
        vkCreateBuffer,
        vkGetBufferMemoryRequirements,
        vkGetPhysicalDeviceMemoryProperties,
    )
    from vulkan_compute.compute_buffer import _find_memory_type

    buffer_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
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
    return buffer, memory, mem_reqs.size


def _write_buffer(device, memory, offset: int, data: np.ndarray) -> None:
    from vulkan import vkMapMemory, vkUnmapMemory
    from vulkan_compute.compute_buffer import _mapped_buffer

    data_bytes = data.view(np.uint8)
    mapped = vkMapMemory(device, memory, offset, data_bytes.nbytes, 0)
    mapped_buf = _mapped_buffer(mapped, data_bytes.nbytes)
    mapped_buf[:] = data_bytes
    vkUnmapMemory(device, memory)


def _write_buffer_bytes(device, memory, offset: int, data: bytes) -> None:
    arr = np.frombuffer(data, dtype=np.uint8)
    _write_buffer(device, memory, offset, arr)


def _read_buffer(device, memory, size: int) -> np.ndarray:
    from vulkan import vkMapMemory, vkUnmapMemory
    from vulkan_compute.compute_buffer import _mapped_buffer

    mapped = vkMapMemory(device, memory, 0, size, 0)
    mapped_buf = _mapped_buffer(mapped, size)
    out = mapped_buf.copy()
    vkUnmapMemory(device, memory)
    return out


def _run_vulkan_tape(
    *,
    prices: np.ndarray,
    volumes: Optional[np.ndarray],
    out_path: pathlib.Path,
    w1: int,
    w2: int,
    eps: float,
    nan_squash: float,
    flags: int,
    shader_path: pathlib.Path,
    spv_path: pathlib.Path,
    vk_icd: Optional[str],
    fp64_returns: bool,
    timing_debug: bool = False,
) -> dict[str, float] | None:
    host_start = time.perf_counter()
    if w2 > 1024:
        raise ValueError("w2 exceeds shader MAX_WINDOW=1024")
    if vk_icd:
        os.environ["VK_ICD_FILENAMES"] = vk_icd

    import vulkan as vk
    from vulkan_compute.compute_buffer import _select_physical_device, _find_queue_family_index

    prices = np.asarray(prices, dtype=np.float32, order="C")
    host_start = time.perf_counter()
    if timing_debug:
        print(f"[vk_qfeat] starting host setup (T={prices.shape[1]}, S={prices.shape[0]})...")
    S, T = prices.shape
    volumes_arr = None
    if volumes is not None:
        volumes_arr = np.asarray(volumes, dtype=np.float32, order="C")

    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vk_qfeat",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="none",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    )
    instance_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )
    instance = vk.vkCreateInstance(instance_info, None)

    physical_device = _select_physical_device(instance)
    queue_family_index = _find_queue_family_index(physical_device)
    enabled_features = None
    if fp64_returns:
        features = vk.vkGetPhysicalDeviceFeatures(physical_device)
        use_fp64_returns = bool(getattr(features, "shaderFloat64", False))
        if use_fp64_returns:
            spv_path = spv_path.with_name(f"{spv_path.stem}.fp64{spv_path.suffix}")
            if timing_debug:
                print(f"[vk_qfeat] compiling shader (fp64) to {spv_path}...")
            _compile_shader(shader_path, spv_path, defines=["USE_FP64_RETURNS=1"])
            enabled_features = vk.VkPhysicalDeviceFeatures(shaderFloat64=vk.VK_TRUE)
        else:
            if timing_debug:
                print(f"[vk_qfeat] compiling shader (fp32 fallback) to {spv_path}...")
            _compile_shader(shader_path, spv_path)
    else:
        if timing_debug:
            print(f"[vk_qfeat] compiling shader to {spv_path}...")
        _compile_shader(shader_path, spv_path)

    if timing_debug:
        print(f"[vk_qfeat] creating device and buffers...")

    queue_info = vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    if enabled_features is None:
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
    else:
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            pEnabledFeatures=enabled_features,
        )
    device = vk.vkCreateDevice(physical_device, device_info, None)
    queue = vk.vkGetDeviceQueue(device, queue_family_index, 0)

    params_buf, params_mem, _ = _create_buffer(device, physical_device, 64)
    price_buf, price_mem, _ = _create_buffer(device, physical_device, prices.nbytes)
    if volumes_arr is not None:
        volume_buf, volume_mem, _ = _create_buffer(device, physical_device, volumes_arr.nbytes)
    else:
        volume_buf, volume_mem, _ = _create_buffer(device, physical_device, 4)
    qfeat_size = S * T * 8 * 4
    qfeat_buf, qfeat_mem, _ = _create_buffer(device, physical_device, qfeat_size)
    dbg_buf, dbg_mem, _ = _create_buffer(device, physical_device, 4)
    weights_buf, weights_mem, weights_size = _create_buffer(
        device, physical_device, ctypes.sizeof(WeightsSSBO)
    )

    meta0 = np.array([S, T, w1, w2], dtype=np.uint32)
    meta1 = np.array([T, T, flags, 0], dtype=np.uint32)
    fmeta0 = np.array([eps, nan_squash, 0.0, 0.0], dtype=np.float32)
    fmeta1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _write_buffer(device, params_mem, 0, meta0)
    _write_buffer(device, params_mem, 16, meta1)
    _write_buffer(device, params_mem, 32, fmeta0)
    _write_buffer(device, params_mem, 48, fmeta1)

    _write_buffer(device, price_mem, 0, prices)
    if volumes_arr is not None:
        _write_buffer(device, volume_mem, 0, volumes_arr)
    weights = _default_weights_ssbo(eps=eps)
    weights_bytes = ctypes.string_at(ctypes.addressof(weights), ctypes.sizeof(weights))
    _write_buffer_bytes(device, weights_mem, 0, weights_bytes)

    shader_code = spv_path.read_bytes()
    shader_info = vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(shader_code),
        pCode=shader_code,
    )
    shader_module = vk.vkCreateShaderModule(device, shader_info, None)

    bindings = []
    for binding in range(6):
        bindings.append(
            vk.VkDescriptorSetLayoutBinding(
                binding=binding,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
        )
    descriptor_set_layout_info = vk.VkDescriptorSetLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=len(bindings),
        pBindings=bindings,
    )
    descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, descriptor_set_layout_info, None)

    pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1,
        pSetLayouts=[descriptor_set_layout],
    )
    pipeline_layout = vk.vkCreatePipelineLayout(device, pipeline_layout_info, None)

    stage_info = vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName="main",
    )
    pipeline_info = vk.VkComputePipelineCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=stage_info,
        layout=pipeline_layout,
    )
    pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

    pool_size = vk.VkDescriptorPoolSize(
        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=6,
    )
    descriptor_pool_info = vk.VkDescriptorPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        poolSizeCount=1,
        pPoolSizes=[pool_size],
        maxSets=1,
    )
    descriptor_pool = vk.vkCreateDescriptorPool(device, descriptor_pool_info, None)
    descriptor_set_alloc = vk.VkDescriptorSetAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=descriptor_pool,
        descriptorSetCount=1,
        pSetLayouts=[descriptor_set_layout],
    )
    descriptor_set = vk.vkAllocateDescriptorSets(device, descriptor_set_alloc)[0]

    buffer_infos = [
        vk.VkDescriptorBufferInfo(buffer=params_buf, offset=0, range=64),
        vk.VkDescriptorBufferInfo(buffer=price_buf, offset=0, range=prices.nbytes),
        vk.VkDescriptorBufferInfo(
            buffer=volume_buf,
            offset=0,
            range=4 if volumes_arr is None else volumes_arr.nbytes,
        ),
        vk.VkDescriptorBufferInfo(buffer=qfeat_buf, offset=0, range=qfeat_size),
        vk.VkDescriptorBufferInfo(buffer=dbg_buf, offset=0, range=4),
        vk.VkDescriptorBufferInfo(buffer=weights_buf, offset=0, range=weights_size),
    ]
    writes = []
    for binding, info in enumerate(buffer_infos):
        writes.append(
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=binding,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[info],
            )
        )
    vk.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

    command_pool_info = vk.VkCommandPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
    )
    command_pool = vk.vkCreateCommandPool(device, command_pool_info, None)
    command_buffer_alloc = vk.VkCommandBufferAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=command_pool,
        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    command_buffer = vk.vkAllocateCommandBuffers(device, command_buffer_alloc)[0]

    begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vk.vkBeginCommandBuffer(command_buffer, begin_info)
    vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(
        command_buffer,
        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        [descriptor_set],
        0,
        None,
    )
    group_count_x = (S + 63) // 64
    vk.vkCmdDispatch(command_buffer, group_count_x, T, 1)
    vk.vkEndCommandBuffer(command_buffer)

    submit_info = vk.VkSubmitInfo(
        sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[command_buffer],
    )
    gpu_start = time.perf_counter()
    vk.vkQueueSubmit(queue, 1, [submit_info], vk.VK_NULL_HANDLE)
    vk.vkQueueWaitIdle(queue)
    gpu_end = time.perf_counter()

    out_bytes = _read_buffer(device, qfeat_mem, qfeat_size)
    mm = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(S, T, 8))
    mm_bytes = mm.view(np.uint8).reshape(-1)
    mm_bytes[:] = out_bytes
    mm.flush()

    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
    vk.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    vk.vkDestroyShaderModule(device, shader_module, None)
    vk.vkDestroyBuffer(device, params_buf, None)
    vk.vkDestroyBuffer(device, price_buf, None)
    vk.vkDestroyBuffer(device, volume_buf, None)
    vk.vkDestroyBuffer(device, qfeat_buf, None)
    vk.vkDestroyBuffer(device, dbg_buf, None)
    vk.vkDestroyBuffer(device, weights_buf, None)
    vk.vkFreeMemory(device, params_mem, None)
    vk.vkFreeMemory(device, price_mem, None)
    vk.vkFreeMemory(device, volume_mem, None)
    vk.vkFreeMemory(device, qfeat_mem, None)
    vk.vkFreeMemory(device, dbg_mem, None)
    vk.vkFreeMemory(device, weights_mem, None)
    vk.vkDestroyCommandPool(device, command_pool, None)
    vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)
    host_end = time.perf_counter()
    if timing_debug:
        return {
            "host_setup": gpu_start - host_start,
            "gpu_compute": gpu_end - gpu_start,
            "host_teardown": host_end - gpu_end,
        }
    return None
