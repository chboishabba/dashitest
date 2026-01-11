from __future__ import annotations

import glob
import os
import struct
from pathlib import Path

import numpy as np
from vulkan import *
from vulkan import ffi


def _load_spirv(path: Path) -> bytes:
    return path.read_bytes()


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


def _mapped_buffer(mapped: object, size: int, dtype: np.dtype) -> np.ndarray:
    try:
        return np.frombuffer(mapped, dtype=dtype, count=size // np.dtype(dtype).itemsize)
    except (TypeError, ValueError):
        try:
            buf = ffi.buffer(mapped, size)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Unsupported mapped buffer type: {type(mapped)}") from exc
        return np.frombuffer(buf, dtype=dtype, count=size // np.dtype(dtype).itemsize)


class VulkanFrameCapture:
    """Headless Vulkan capture of the sheet expand shader as a NumPy frame."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        sheet_w: int,
        sheet_h: int,
        block_px: int = 16,
        alpha: float = 0.97,
        vmin: float = 0.0,
        vmax: float = 1.0,
        use_clamp: bool = False,
        shader_path: Path | None = None,
    ) -> None:
        if shader_path is None:
            shader_path = Path(__file__).parent / "shaders" / "sheet_expand_fade.spv"
        if not shader_path.exists():
            raise FileNotFoundError(f"{shader_path} not found; compile it with glslc.")
        self.width = width
        self.height = height
        self.sheet_w = sheet_w
        self.sheet_h = sheet_h
        self.block_px = block_px
        self.alpha = float(alpha)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.use_clamp = use_clamp
        self.shader_path = shader_path
        self._instance: VkInstance | None = None
        self._device: VkDevice | None = None
        self._queue: VkQueue | None = None
        self._command_pool: VkCommandPool | None = None
        self._command_buffer: VkCommandBuffer | None = None
        self._pipeline_layout: VkPipelineLayout | None = None
        self._compute_pipeline: VkPipeline | None = None
        self._descriptor_pool: VkDescriptorPool | None = None
        self._descriptor_set_layout: VkDescriptorSetLayout | None = None
        self._descriptor_set: VkDescriptorSet | None = None
        self._shader_module: VkShaderModule | None = None
        self._compute_image: VkImage | None = None
        self._compute_image_view: VkImageView | None = None
        self._compute_memory: VkDeviceMemory | None = None
        self._accum_image: VkImage | None = None
        self._accum_view: VkImageView | None = None
        self._accum_memory: VkDeviceMemory | None = None
        self._sheet_buffer: VkBuffer | None = None
        self._sheet_memory: VkDeviceMemory | None = None
        self._sheet_mapped: object | None = None
        self._sheet_data: np.ndarray | None = None
        self._record_buffer: VkBuffer | None = None
        self._record_memory: VkDeviceMemory | None = None
        self._record_size = self.width * self.height * 4
        self._record_mapped: object | None = None
        self._record_memory_range: VkMappedMemoryRange | None = None
        self._pc_data = struct.pack(
            "<5I3fI",
            self.sheet_w,
            self.sheet_h,
            self.width,
            self.height,
            self.block_px,
            self.alpha,
            self.vmin,
            self.vmax,
            1 if self.use_clamp else 0,
        )
        self._pc_buffer = ffi.new("char[]", self._pc_data)
        self._mem_props: VkPhysicalDeviceMemoryProperties | None = None
        self._closed = False
        self._init_vulkan()

    def __enter__(self) -> "VulkanFrameCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _init_vulkan(self) -> None:
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="vulkan_frame_capture",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="none",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 0, 0),
        )
        instance_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        self._instance = vkCreateInstance(instance_info, None)
        physical_device = _select_physical_device(self._instance)
        queue_family_index = _find_compute_queue_family_index(physical_device)
        queue_priority = 1.0
        device_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[
                VkDeviceQueueCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=queue_family_index,
                    queueCount=1,
                    pQueuePriorities=[queue_priority],
                )
            ],
        )
        self._device = vkCreateDevice(physical_device, device_info, None)
        self._queue = vkGetDeviceQueue(self._device, queue_family_index, 0)
        self._mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)

        self._create_images()
        self._create_sheet_buffer()
        self._create_staging_buffer()
        self._create_pipeline()
        self._create_descriptor_sets()
        self._create_command_buffer(queue_family_index)

    def _create_images(self) -> None:
        if self._device is None:
            raise RuntimeError("Device not initialized")
        image_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=VK_IMAGE_TYPE_2D,
            format=VK_FORMAT_R8G8B8A8_UNORM,
            extent=VkExtent3D(width=self.width, height=self.height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._compute_image = vkCreateImage(self._device, image_info, None)
        img_reqs = vkGetImageMemoryRequirements(self._device, self._compute_image)
        img_type = _find_memory_type(
            self._mem_props,
            img_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=img_reqs.size,
            memoryTypeIndex=img_type,
        )
        self._compute_memory = vkAllocateMemory(self._device, alloc_info, None)
        vkBindImageMemory(self._device, self._compute_image, self._compute_memory, 0)
        view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self._compute_image,
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
        self._compute_image_view = vkCreateImageView(self._device, view_info, None)

        accum_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=VK_IMAGE_TYPE_2D,
            format=VK_FORMAT_R32_SFLOAT,
            extent=VkExtent3D(width=self.width, height=self.height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._accum_image = vkCreateImage(self._device, accum_info, None)
        accum_reqs = vkGetImageMemoryRequirements(self._device, self._accum_image)
        accum_type = _find_memory_type(
            self._mem_props,
            accum_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        accum_alloc = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=accum_reqs.size,
            memoryTypeIndex=accum_type,
        )
        self._accum_memory = vkAllocateMemory(self._device, accum_alloc, None)
        vkBindImageMemory(self._device, self._accum_image, self._accum_memory, 0)
        accum_view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self._accum_image,
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
        self._accum_view = vkCreateImageView(self._device, accum_view_info, None)

    def _create_sheet_buffer(self) -> None:
        if self._device is None:
            raise RuntimeError("Device not initialized")
        sheet_size = self.sheet_w * self.sheet_h * 4
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=sheet_size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        self._sheet_buffer = vkCreateBuffer(self._device, buffer_info, None)
        reqs = vkGetBufferMemoryRequirements(self._device, self._sheet_buffer)
        mem_type = _find_memory_type(
            self._mem_props,
            reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type,
        )
        self._sheet_memory = vkAllocateMemory(self._device, alloc_info, None)
        vkBindBufferMemory(self._device, self._sheet_buffer, self._sheet_memory, 0)
        self._sheet_mapped = vkMapMemory(self._device, self._sheet_memory, 0, sheet_size, 0)
        self._sheet_data = _mapped_buffer(self._sheet_mapped, sheet_size, np.float32)

    def _create_staging_buffer(self) -> None:
        if self._device is None:
            raise RuntimeError("Device not initialized")
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=self._record_size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        self._record_buffer = vkCreateBuffer(self._device, buffer_info, None)
        reqs = vkGetBufferMemoryRequirements(self._device, self._record_buffer)
        mem_type = _find_memory_type(
            self._mem_props,
            reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type,
        )
        self._record_memory = vkAllocateMemory(self._device, alloc_info, None)
        vkBindBufferMemory(self._device, self._record_buffer, self._record_memory, 0)
        self._record_mapped = vkMapMemory(self._device, self._record_memory, 0, self._record_size, 0)
        self._record_memory_range = VkMappedMemoryRange(
            sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            memory=self._record_memory,
            offset=0,
            size=self._record_size,
        )

    def _create_pipeline(self) -> None:
        if self._device is None:
            raise RuntimeError("Device not initialized")
        code = _load_spirv(self.shader_path)
        self._shader_module = vkCreateShaderModule(
            self._device,
            VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code,
            ),
            None,
        )
        bindings = [
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
        self._descriptor_set_layout = vkCreateDescriptorSetLayout(
            self._device,
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
            size=len(self._pc_data),
        )
        self._pipeline_layout = vkCreatePipelineLayout(
            self._device,
            VkPipelineLayoutCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self._descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_range],
            ),
            None,
        )
        stage = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self._shader_module,
            pName=ffi.new("char[]", b"main"),
        )
        pipeline_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=self._pipeline_layout,
        )
        self._compute_pipeline = vkCreateComputePipelines(
            self._device,
            VK_NULL_HANDLE,
            1,
            [pipeline_info],
            None,
        )[0]

    def _create_descriptor_sets(self) -> None:
        if self._device is None or self._descriptor_set_layout is None:
            raise RuntimeError("Pipeline not ready")
        pool_sizes = [
            VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1),
            VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=2),
        ]
        self._descriptor_pool = vkCreateDescriptorPool(
            self._device,
            VkDescriptorPoolCreateInfo(
                sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes,
                maxSets=1,
            ),
            None,
        )
        self._descriptor_set = vkAllocateDescriptorSets(
            self._device,
            VkDescriptorSetAllocateInfo(
                sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self._descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self._descriptor_set_layout],
            ),
        )[0]
        writes = [
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._descriptor_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    VkDescriptorBufferInfo(
                        buffer=self._sheet_buffer,
                        offset=0,
                        range=self.sheet_w * self.sheet_h * 4,
                    )
                ],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._descriptor_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[
                    VkDescriptorImageInfo(
                        imageView=self._accum_view,
                        imageLayout=VK_IMAGE_LAYOUT_GENERAL,
                    )
                ],
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._descriptor_set,
                dstBinding=2,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[
                    VkDescriptorImageInfo(
                        imageView=self._compute_image_view,
                        imageLayout=VK_IMAGE_LAYOUT_GENERAL,
                    )
                ],
            ),
        ]
        vkUpdateDescriptorSets(self._device, len(writes), writes, 0, None)

    def _create_command_buffer(self, queue_family_index: int) -> None:
        if self._device is None:
            raise RuntimeError("Device not initialized")
        self._command_pool = vkCreateCommandPool(
            self._device,
            VkCommandPoolCreateInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=queue_family_index,
            ),
            None,
        )
        self._command_buffer = vkAllocateCommandBuffers(
            self._device,
            VkCommandBufferAllocateInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self._command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]

    def capture(self, sheet_values: np.ndarray | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("VulkanFrameCapture is closed")
        if self._command_buffer is None:
            raise RuntimeError("Command buffer not initialized")
        if sheet_values is not None:
            self._write_sheet(sheet_values)
        else:
            if self._sheet_data is not None:
                self._sheet_data.fill(0.0)
        cmd = self._command_buffer
        vkResetCommandBuffer(cmd, 0)
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))

        image_barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=VK_IMAGE_LAYOUT_GENERAL,
            image=self._compute_image,
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
            [image_barrier],
        )

        accum_barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=VK_IMAGE_LAYOUT_GENERAL,
            image=self._accum_image,
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
            [accum_barrier],
        )

        clear_color = VkClearColorValue(float32=[0.0, 0.0, 0.0, 0.0])
        vkCmdClearColorImage(
            cmd,
            self._accum_image,
            VK_IMAGE_LAYOUT_GENERAL,
            clear_color,
            1,
            [
                VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                )
            ],
        )

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self._compute_pipeline)
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self._pipeline_layout,
            0,
            1,
            [self._descriptor_set],
            0,
            None,
        )
        vkCmdPushConstants(
            cmd,
            self._pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            len(self._pc_data),
            self._pc_buffer,
        )
        vkCmdDispatch(cmd, (self.width + 15) // 16, (self.height + 15) // 16, 1)

        transfer_barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=VK_IMAGE_LAYOUT_GENERAL,
            newLayout=VK_IMAGE_LAYOUT_GENERAL,
            image=self._compute_image,
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
            [transfer_barrier],
        )
        vkCmdCopyImageToBuffer(
            cmd,
            self._compute_image,
            VK_IMAGE_LAYOUT_GENERAL,
            self._record_buffer,
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
                    imageExtent=VkExtent3D(width=self.width, height=self.height, depth=1),
                )
            ],
        )
        vkEndCommandBuffer(cmd)

        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vkQueueSubmit(self._queue, 1, [submit], VK_NULL_HANDLE)
        vkQueueWaitIdle(self._queue)

        if self._record_memory_range is not None:
            vkInvalidateMappedMemoryRanges(self._device, 1, [self._record_memory_range])

        buf = ffi.buffer(self._record_mapped, self._record_size)
        frame = np.frombuffer(buf, dtype=np.uint8).copy()
        return frame.reshape(self.height, self.width, 4)

    def _write_sheet(self, values: np.ndarray) -> None:
        if self._sheet_data is None:
            raise RuntimeError("Sheet buffer is not mapped")
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (self.sheet_h, self.sheet_w):
            raise ValueError(f"Sheet values must be shape ({self.sheet_h}, {self.sheet_w})")
        self._sheet_data[:] = arr.ravel()

    def close(self) -> None:
        if self._closed:
            return
        if self._device:
            vkDeviceWaitIdle(self._device)
        if self._sheet_buffer and self._sheet_mapped:
            vkUnmapMemory(self._device, self._sheet_memory)
        if self._record_buffer and self._record_mapped:
            vkUnmapMemory(self._device, self._record_memory)
        if self._command_pool and self._command_buffer:
            vkFreeCommandBuffers(self._device, self._command_pool, 1, [self._command_buffer])
        if self._command_pool:
            vkDestroyCommandPool(self._device, self._command_pool, None)
        if self._compute_pipeline:
            vkDestroyPipeline(self._device, self._compute_pipeline, None)
        if self._pipeline_layout:
            vkDestroyPipelineLayout(self._device, self._pipeline_layout, None)
        if self._descriptor_pool:
            vkDestroyDescriptorPool(self._device, self._descriptor_pool, None)
        if self._descriptor_set_layout:
            vkDestroyDescriptorSetLayout(self._device, self._descriptor_set_layout, None)
        if self._shader_module:
            vkDestroyShaderModule(self._device, self._shader_module, None)
        if self._sheet_buffer:
            vkDestroyBuffer(self._device, self._sheet_buffer, None)
        if self._sheet_memory:
            vkFreeMemory(self._device, self._sheet_memory, None)
        if self._record_buffer:
            vkDestroyBuffer(self._device, self._record_buffer, None)
        if self._record_memory:
            vkFreeMemory(self._device, self._record_memory, None)
        if self._compute_image_view:
            vkDestroyImageView(self._device, self._compute_image_view, None)
        if self._compute_image:
            vkDestroyImage(self._device, self._compute_image, None)
        if self._compute_memory:
            vkFreeMemory(self._device, self._compute_memory, None)
        if self._accum_view:
            vkDestroyImageView(self._device, self._accum_view, None)
        if self._accum_image:
            vkDestroyImage(self._device, self._accum_image, None)
        if self._accum_memory:
            vkFreeMemory(self._device, self._accum_memory, None)
        if self._device:
            vkDestroyDevice(self._device, None)
        if self._instance:
            vkDestroyInstance(self._instance, None)
        self._closed = True
