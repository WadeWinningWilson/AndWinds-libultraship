#pragma once

/* Plain-C form of the host Aurora a hosted libultraship draws through (gfx_aurora_host.h), for frontends whose
   host hands them its renderer across a C boundary (2Ship's hosted provider, filled by ALBT&W's .dusk adapter from
   the host's gfx service, svc/gfx.h). WebGPU handles are the host's own (WGPUDevice etc.), passed as void* so this
   header needs no WebGPU headers.

   Threads: register_* and push_* are called from libultraship's game thread while it records a frame. A host
   whose service must run on its own thread queues them and issues them there; libultraship keeps 4 frames of
   state, so a push may land up to one host frame later. The callbacks run on the host's render worker. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LUS_AURORA_HOST_VERSION 1u
#define LUS_AURORA_MAX_COLOR_ATTACHMENTS 8u

typedef struct LusAuroraTargetLayout {
    uint64_t key;                 /* changes when the layout does (rebuild pipelines) */
    uint32_t color_count;         /* scene colour is attachment 0 */
    uint32_t color_formats[LUS_AURORA_MAX_COLOR_ATTACHMENTS]; /* WGPUTextureFormat */
    uint32_t depth_format;        /* WGPUTextureFormat, 0 = none */
    uint32_t sample_count;
    uint32_t width;               /* scene colour size */
    uint32_t height;
    uint32_t reversed_z;          /* 1: depth 1.0 is near */
} LusAuroraTargetLayout;

/* Inside the host's scene pass. */
typedef struct LusAuroraDrawContext {
    void* device; /* WGPUDevice */
    void* queue;  /* WGPUQueue */
    void* pass;   /* WGPURenderPassEncoder */
    LusAuroraTargetLayout layout;
} LusAuroraDrawContext;
typedef void (*LusAuroraDrawFn)(void* fn_user, const LusAuroraDrawContext* ctx, const void* payload, size_t size);

/* On the host's frame encoder, between scene passes: no pass left open, never finished. */
typedef void (*LusAuroraTaskFn)(void* fn_user, void* device, void* queue, void* encoder, const void* payload,
                                size_t size);

/* A world drawn in the host's scene pass shares the host's depth (ALBT&W HR10-Q4): framebuffer 0 is composited with
   its depth converted to the host camera's. proj_z = the host projection's z and w rows along the view axis
   (column-major proj_from_view [10], [14], [11], [15]: depth = (p0 * z + p1) / (p2 * z + p3) for view z < 0), in host
   units; world_scale = libultraship's world units per host unit. */
typedef struct LusAuroraWorldDepth {
    uint32_t enabled; /* 0: composite without depth (full-screen play) */
    float proj_z[4];
    float world_scale;
} LusAuroraWorldDepth;

typedef struct LusAuroraHostV1 {
    uint32_t struct_size;
    uint32_t version; /* LUS_AURORA_HOST_VERSION */
    void* user;       /* passed back to every function below */
    /* The host's wgpu* functions by name ("wgpuDeviceCreateBuffer"). */
    void* (*get_proc)(void* user, const char* name);
    void* device; /* WGPUDevice, borrowed */
    void* queue;  /* WGPUQueue, borrowed */
    /* The host's scene target now. 1 = filled. */
    int (*scene_layout)(void* user, LusAuroraTargetLayout* out);
    uint64_t (*register_draw)(void* user, const char* label, LusAuroraDrawFn fn, void* fn_user); /* 0 = failed */
    void (*unregister_draw)(void* user, uint64_t handle);
    int (*push_draw)(void* user, uint64_t handle, const void* payload, size_t size); /* 1 = queued */
    uint64_t (*register_task)(void* user, const char* label, LusAuroraTaskFn fn, void* fn_user);
    void (*unregister_task)(void* user, uint64_t handle);
    int (*push_task)(void* user, uint64_t handle, const void* payload, size_t size);
    /* Optional (struct_size covers it): the host camera's depth for the composite, asked on the render worker as
       framebuffer 0 is drawn. 1 = filled. */
    int (*world_depth)(void* user, LusAuroraWorldDepth* out);
} LusAuroraHostV1;
#define LUS_AURORA_HOST_V1_BASE_SIZE offsetof(LusAuroraHostV1, world_depth)

#ifdef __cplusplus
}

namespace Fast {
// Installs a host's Aurora for the next window libultraship creates (copies the table; nullptr restores the bridge).
// Fills libultraship's WebGPU from host->get_proc first (builds with LUS_AURORA_HOST_PROCS). false: not usable
// (wrong version, functions missing, or this build has no Aurora backend).
bool SetAuroraGfxHostC(const LusAuroraHostV1* host);
// True when this build can draw with a host's device (the Aurora backend with LUS_AURORA_HOST_PROCS).
bool AuroraGfxHostCSupported();
// The game's own view planes for the frame being drawn (MM: play->view zNear / zFar), so its depth can be turned back
// into distance for the host's depth. Any thread; 0 / 0 = unknown (the composite then writes no depth).
void SetAuroraWorldDepthRange(float zNear, float zFar);
} // namespace Fast
#endif
