#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <va/va.h>
#include <va/va_drm.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/hwcontext_vaapi.h>

#define MAX_OBJECTS 4
#define MAX_PLANES 4

struct DmabufObjectPacked {
    uint64_t modifier;
    uint64_t size;
} __attribute__((packed));

struct DmabufPlanePacked {
    uint32_t object_index;
    uint32_t offset;
    uint32_t pitch;
} __attribute__((packed));

struct DmabufInfoPacked {
    uint32_t width;
    uint32_t height;
    uint32_t drm_format;
    uint32_t nb_objects;
    uint32_t nb_planes;
    struct DmabufObjectPacked objects[MAX_OBJECTS];
    struct DmabufPlanePacked planes[MAX_PLANES];
} __attribute__((packed));

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    while (*pix_fmts != AV_PIX_FMT_NONE) {
        if (*pix_fmts == AV_PIX_FMT_VAAPI) {
            return *pix_fmts;
        }
        pix_fmts++;
    }
    fprintf(stderr, "vaapi dmabuf export: VAAPI pixel format not offered by decoder\n");
    return AV_PIX_FMT_NONE;
}

static int send_dmabuf_info(int sock_fd, const AVFrame *drm_frame) {
    const AVDRMFrameDescriptor *desc = (const AVDRMFrameDescriptor *)drm_frame->data[0];
    if (!desc) {
        fprintf(stderr, "vaapi dmabuf export: missing DRM descriptor\n");
        return -1;
    }
    if (desc->nb_objects > MAX_OBJECTS) {
        fprintf(stderr, "vaapi dmabuf export: too many objects (%d > %d)\n", desc->nb_objects, MAX_OBJECTS);
        return -1;
    }
    int plane_count = 0;
    for (int i = 0; i < desc->nb_layers; i++) {
        plane_count += desc->layers[i].nb_planes;
    }
    if (plane_count > MAX_PLANES) {
        fprintf(stderr, "vaapi dmabuf export: too many planes (%d > %d)\n", plane_count, MAX_PLANES);
        return -1;
    }

    struct DmabufInfoPacked info;
    memset(&info, 0, sizeof(info));
    info.width = (uint32_t)drm_frame->width;
    info.height = (uint32_t)drm_frame->height;
    info.drm_format = desc->nb_layers > 0 ? desc->layers[0].format : 0;
    info.nb_objects = (uint32_t)desc->nb_objects;
    info.nb_planes = (uint32_t)plane_count;

    for (int i = 0; i < desc->nb_objects; i++) {
        info.objects[i].modifier = desc->objects[i].format_modifier;
        info.objects[i].size = (uint64_t)desc->objects[i].size;
    }

    int plane_index = 0;
    for (int i = 0; i < desc->nb_layers; i++) {
        for (int j = 0; j < desc->layers[i].nb_planes; j++) {
            if (plane_index >= MAX_PLANES) {
                break;
            }
            info.planes[plane_index].object_index = desc->layers[i].planes[j].object_index;
            info.planes[plane_index].offset = desc->layers[i].planes[j].offset;
            info.planes[plane_index].pitch = desc->layers[i].planes[j].pitch;
            plane_index++;
        }
    }

    int fds[MAX_OBJECTS];
    for (int i = 0; i < desc->nb_objects; i++) {
        fds[i] = desc->objects[i].fd;
    }

    char cmsg_buf[CMSG_SPACE(sizeof(int) * MAX_OBJECTS)];
    memset(cmsg_buf, 0, sizeof(cmsg_buf));

    struct iovec iov = {
        .iov_base = &info,
        .iov_len = sizeof(info),
    };
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * desc->nb_objects);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * desc->nb_objects);
    memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * desc->nb_objects);

    ssize_t sent = sendmsg(sock_fd, &msg, 0);
    if (sent < 0) {
        fprintf(stderr, "vaapi dmabuf export: sendmsg failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video> [vaapi_device] [--force-linear] [--debug] [--drm-device <path>]\n", argv[0]);
        return 1;
    }
    const char *sock_env = getenv("DMABUF_STUB_SOCK_FD");
    if (!sock_env) {
        fprintf(stderr, "vaapi dmabuf export: DMABUF_STUB_SOCK_FD not set\n");
        return 1;
    }
    int sock_fd = atoi(sock_env);
    if (sock_fd <= 0) {
        fprintf(stderr, "vaapi dmabuf export: invalid socket fd\n");
        return 1;
    }
    const char *input = argv[1];
    const char *vaapi_device = "/dev/dri/renderD128";
    const char *drm_device = NULL;
    int force_linear = 0;
    int debug = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--force-linear") == 0) {
            force_linear = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = 1;
        } else if (strcmp(argv[i], "--drm-device") == 0 && i + 1 < argc) {
            drm_device = argv[i + 1];
            i++;
        } else {
            vaapi_device = argv[i];
        }
    }
    if (!drm_device) {
        drm_device = vaapi_device;
    }
    if (debug) {
        fprintf(stderr, "vaapi dmabuf export: input=%s device=%s force_linear=%d drm_device=%s\n", input, vaapi_device, force_linear, drm_device);
    }

    av_log_set_level(AV_LOG_ERROR);

    AVFormatContext *fmt_ctx = NULL;
    if (avformat_open_input(&fmt_ctx, input, NULL, NULL) < 0) {
        fprintf(stderr, "vaapi dmabuf export: failed to open input\n");
        return 1;
    }
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "vaapi dmabuf export: failed to read stream info\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    int video_stream = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = (int)i;
            break;
        }
    }
    if (video_stream < 0) {
        fprintf(stderr, "vaapi dmabuf export: no video stream found\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    const AVCodec *decoder = avcodec_find_decoder(fmt_ctx->streams[video_stream]->codecpar->codec_id);
    if (!decoder) {
        fprintf(stderr, "vaapi dmabuf export: decoder not found\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        fprintf(stderr, "vaapi dmabuf export: failed to allocate codec context\n");
        avformat_close_input(&fmt_ctx);
        return 1;
    }
    if (avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream]->codecpar) < 0) {
        fprintf(stderr, "vaapi dmabuf export: failed to copy codec params\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return 1;
    }
    codec_ctx->get_format = get_hw_format;

    AVBufferRef *hw_device_ctx = NULL;
    int va_fd = -1;
    VADisplay va_display = NULL;
    int ret_hw = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_VAAPI, vaapi_device, NULL, 0);
    if (ret_hw < 0) {
        if (debug) {
            fprintf(stderr, "vaapi dmabuf export: av_hwdevice_ctx_create failed, trying vaGetDisplayDRM\n");
        }
        va_fd = open(vaapi_device, O_RDWR);
        if (va_fd < 0) {
            fprintf(stderr, "vaapi dmabuf export: failed to open %s: %s\n", vaapi_device, strerror(errno));
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
        if (fcntl(va_fd, F_SETFD, FD_CLOEXEC) == -1) {
            fprintf(stderr, "vaapi dmabuf export: failed to set CLOEXEC: %s\n", strerror(errno));
        }
        va_display = vaGetDisplayDRM(va_fd);
        if (!va_display) {
            fprintf(stderr, "vaapi dmabuf export: vaGetDisplayDRM failed\n");
            close(va_fd);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
        hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VAAPI);
        if (!hw_device_ctx) {
            fprintf(stderr, "vaapi dmabuf export: failed to allocate VAAPI hwdevice ctx\n");
            close(va_fd);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
        AVHWDeviceContext *hwdev = (AVHWDeviceContext *)hw_device_ctx->data;
        AVVAAPIDeviceContext *va_ctx = (AVVAAPIDeviceContext *)hwdev->hwctx;
        va_ctx->display = va_display;
        if (av_hwdevice_ctx_init(hw_device_ctx) < 0) {
            fprintf(stderr, "vaapi dmabuf export: failed to init VAAPI hwdevice ctx\n");
            av_buffer_unref(&hw_device_ctx);
            vaTerminate(va_display);
            close(va_fd);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
    }
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    if (avcodec_open2(codec_ctx, decoder, NULL) < 0) {
        fprintf(stderr, "vaapi dmabuf export: failed to open decoder\n");
        av_buffer_unref(&hw_device_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return 1;
    }
    if (debug) {
        fprintf(stderr, "vaapi dmabuf export: decoder opened (%dx%d)\n", codec_ctx->width, codec_ctx->height);
    }

    AVBufferRef *drm_device_ctx = NULL;
    AVBufferRef *drm_frames_ctx = NULL;
    if (force_linear) {
        if (av_hwdevice_ctx_create(&drm_device_ctx, AV_HWDEVICE_TYPE_DRM, drm_device, NULL, 0) < 0) {
            fprintf(stderr, "vaapi dmabuf export: failed to create DRM device (%s)\n", drm_device);
            av_buffer_unref(&hw_device_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
        drm_frames_ctx = av_hwframe_ctx_alloc(drm_device_ctx);
        if (!drm_frames_ctx) {
            fprintf(stderr, "vaapi dmabuf export: failed to alloc DRM frames ctx\n");
            av_buffer_unref(&drm_device_ctx);
            av_buffer_unref(&hw_device_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
        AVHWFramesContext *frames_ctx = (AVHWFramesContext *)drm_frames_ctx->data;
        frames_ctx->format = AV_PIX_FMT_DRM_PRIME;
        frames_ctx->sw_format = AV_PIX_FMT_NV12;
        frames_ctx->width = codec_ctx->width;
        frames_ctx->height = codec_ctx->height;
        frames_ctx->initial_pool_size = 1;
        if (av_hwframe_ctx_init(drm_frames_ctx) < 0) {
            fprintf(stderr, "vaapi dmabuf export: failed to init DRM frames ctx\n");
            av_buffer_unref(&drm_frames_ctx);
            av_buffer_unref(&drm_device_ctx);
            av_buffer_unref(&hw_device_ctx);
            avcodec_free_context(&codec_ctx);
            avformat_close_input(&fmt_ctx);
            return 1;
        }
    }

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *drm_frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    if (!pkt || !frame || !drm_frame || !sw_frame) {
        fprintf(stderr, "vaapi dmabuf export: allocation failure\n");
        av_packet_free(&pkt);
        av_frame_free(&frame);
        av_frame_free(&drm_frame);
        av_frame_free(&sw_frame);
        av_buffer_unref(&drm_frames_ctx);
        av_buffer_unref(&drm_device_ctx);
        av_buffer_unref(&hw_device_ctx);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return 1;
    }

    int sent = 0;
    while (!sent && av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_stream) {
            av_packet_unref(pkt);
            continue;
        }
        int ret = avcodec_send_packet(codec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0) {
            continue;
        }
        while ((ret = avcodec_receive_frame(codec_ctx, frame)) == 0) {
            if (debug) {
                fprintf(stderr, "vaapi dmabuf export: got frame format=%d\n", frame->format);
            }
            if (frame->format != AV_PIX_FMT_VAAPI) {
                fprintf(stderr, "vaapi dmabuf export: decoded frame not VAAPI\n");
                break;
            }
            if (force_linear) {
                sw_frame->format = AV_PIX_FMT_NV12;
                sw_frame->width = frame->width;
                sw_frame->height = frame->height;
                if (av_frame_get_buffer(sw_frame, 1) < 0) {
                    fprintf(stderr, "vaapi dmabuf export: failed to alloc sw frame\n");
                    break;
                }
                if (debug) {
                    fprintf(stderr, "vaapi dmabuf export: downloading to sw NV12\n");
                }
                ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                if (ret < 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    fprintf(stderr, "vaapi dmabuf export: failed to download to sw frame: %s\n", errbuf);
                    av_frame_unref(sw_frame);
                    break;
                }
                drm_frame->format = AV_PIX_FMT_DRM_PRIME;
                drm_frame->width = frame->width;
                drm_frame->height = frame->height;
                drm_frame->hw_frames_ctx = av_buffer_ref(drm_frames_ctx);
                ret = av_hwframe_get_buffer(drm_frames_ctx, drm_frame, 0);
                if (ret < 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    fprintf(stderr, "vaapi dmabuf export: failed to alloc DRM frame: %s\n", errbuf);
                    av_frame_unref(sw_frame);
                    break;
                }
                if (debug) {
                    fprintf(stderr, "vaapi dmabuf export: uploading to DRM frame\n");
                }
                ret = av_hwframe_transfer_data(drm_frame, sw_frame, 0);
                if (ret < 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    fprintf(stderr, "vaapi dmabuf export: failed to upload to DRM frame: %s\n", errbuf);
                    av_frame_unref(sw_frame);
                    av_frame_unref(drm_frame);
                    break;
                }
                av_frame_unref(sw_frame);
            } else {
                drm_frame->format = AV_PIX_FMT_DRM_PRIME;
                drm_frame->width = frame->width;
                drm_frame->height = frame->height;
                if (av_hwframe_map(drm_frame, frame, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT) < 0) {
                    fprintf(stderr, "vaapi dmabuf export: failed to map to DRM PRIME\n");
                    break;
                }
            }

            if (send_dmabuf_info(sock_fd, drm_frame) == 0) {
                sent = 1;
            }
            av_frame_unref(drm_frame);
            av_frame_unref(frame);
            if (sent) {
                break;
            }
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&drm_frame);
    av_frame_free(&sw_frame);
    av_buffer_unref(&drm_frames_ctx);
    av_buffer_unref(&drm_device_ctx);
    av_buffer_unref(&hw_device_ctx);
    if (va_display) {
        vaTerminate(va_display);
    }
    if (va_fd >= 0) {
        close(va_fd);
    }
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    if (!sent) {
        fprintf(stderr, "vaapi dmabuf export: failed to export a frame\n");
        return 1;
    }
    return 0;
}
