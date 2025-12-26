# VAAPI Zero-Copy Path (WIP)

Goal: decode via VAAPI and import GPU frames directly into Vulkan without
`hwdownload` or CPU staging.

## Why it matters

Current pipeline still pulls frames back to CPU for upload. Zero-copy aims to:
- keep decode on GPU (VAAPI)
- import dmabuf/DRM PRIME frames into Vulkan
- run compute/preview without CPU copies

## Expected prerequisites

- ffmpeg build with VAAPI support
- DRM render node (`/dev/dri/renderD128`)
- Vulkan device extensions:
  - `VK_KHR_external_memory_fd`
  - `VK_EXT_external_memory_dma_buf`
  - `VK_EXT_image_drm_format_modifier`

Run the probe:
```
python vulkan/vaapi_probe.py
```

## Proposed flow (high level)

1. Decode with VAAPI to DRM PRIME frames (dmabuf handles)
2. Import dmabuf into Vulkan via `VkImportMemoryFdInfoKHR`
3. Create `VkImage` with DRM format modifier info
4. Bind imported memory to the Vulkan image
5. Run compute + render without CPU staging

## Minimal dmabuf interop stub (next experiment)

Goal: prove zero-copy decode -> Vulkan import works for a single frame. No diff,
no render, no entropy coding.

1) Decode one VAAPI frame as DRM PRIME (dmabuf-backed), not `hwdownload`:
```
ffmpeg \
  -hwaccel vaapi \
  -vaapi_device /dev/dri/renderD128 \
  -hwaccel_output_format drm_prime \
  -i input.mp4 \
  -vf scale_vaapi=w=1920:h=1080 \
  -f rawvideo \
  -pix_fmt drm_prime \
  -frames:v 1 \
  -
```
2) Extract dmabuf metadata (FDs, offsets, strides, format) from the `AVFrame`.
3) Import dmabuf into Vulkan external memory:
   - enable `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`,
     `VK_EXT_image_drm_format_modifier`
   - `VkImage` format likely `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` for NV12
   - `VkImportMemoryFdInfoKHR` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`
4) Bind memory to the image and transition layout to `GENERAL` or
   `SHADER_READ_ONLY_OPTIMAL`.
5) Stop. Success = image import works without CPU copies.

## Stub implementation

`vulkan/vaapi_dmabuf_stub.py` implements the minimal end-to-end test:

- Builds `vulkan/vaapi_dmabuf_export.c` (libavformat/libavcodec/libavutil)
- Decodes one VAAPI frame and exports a dmabuf via `AV_PIX_FMT_DRM_PRIME`
- Imports the dmabuf into Vulkan with the external memory + DRM modifier path
- Performs a single layout transition barrier and exits

Run:
```
python vulkan/vaapi_dmabuf_stub.py path/to/video.mp4 --vaapi-device /dev/dri/renderD128
```
Add a shorter timeout if decode stalls:
```
python vulkan/vaapi_dmabuf_stub.py path/to/video.mp4 --timeout-s 5
```
If modifiers are implicit (e.g. `DRM_FORMAT_MOD_INVALID`), force a linear
dmabuf by downloading to NV12 and re-uploading into a DRM buffer:
```
python vulkan/vaapi_dmabuf_stub.py path/to/video.mp4 --force-linear
```

Notes:
- Currently supports NV12 (`NV12`) and P010 (`P010`) multi-plane formats, plus
  single-plane `R16` when that is what VAAPI exports.
- Multi-object dmabufs are not supported yet.
- If `VK_EXT_image_drm_format_modifier` is missing, try setting
  `VK_ICD_FILENAMES` to your Mesa RADV ICD (the stub will report the device
  and related extensions when it fails).
- Without `VK_EXT_image_drm_format_modifier`, the stub only accepts linear
  dmabufs (modifier 0). Non-linear modifiers will fail. The exporter may report
  `DRM_FORMAT_MOD_INVALID` (all 1s); the stub treats that as linear for the
  minimal experiment.
- Requires Vulkan extensions listed above, plus working VAAPI decode.

## Status

- Probe script exists.
- Minimal dmabuf import stub exists (`vulkan/vaapi_dmabuf_stub.py`).
- Python binding support for external memory structs is available on this setup.

## Next steps

- Confirm device extensions via `vaapi_probe.py`.
- Evaluate Python bindings for external memory structs.
- Prototype importing a single dmabuf into Vulkan.
