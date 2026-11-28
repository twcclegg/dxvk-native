/* Apple port experiment (twcclegg/dxvk-native): true two-eye stereo captures, for comparing against the
 * depth-reprojected stereo of the visionOS compositor. Off unless LB_STEREO_EYE_UNITS is set (non-zero) in the
 * environment before the first vertex shader is translated.
 *
 * When on, every programmable D3D9 vertex shader ends with
 *     pos.x -= k * (abs(pos.w - 1) > 1e-5)      (or pos -= offset, dxvk_stereo_set_clip_offset)
 * so perspective geometry is drawn as seen from a camera moved sideways, and orthographic geometry (sun shadow
 * cascades, post-processing quads, UI: all w == 1) is left alone. k is a clip-space constant: for an eye at
 * view-space x = e (e < 0: left eye) it is k = P00 * e, P00 = 2 / (tan_right - tan_left) of the world camera.
 * Derivation in src/dxso/dxso_compiler.cpp (DxsoCompiler::emitVsClipping).
 *
 * k is applied per draw and only to passes whose viewport has the back buffer's aspect ratio (the main scene),
 * not to square shadow maps of spot/point lights or other off-screen passes; LB_STEREO_PASSES=all applies it to
 * every perspective draw. Fixed-function vertex processing is never shifted.
 *
 * Single-pass stereo (multiview): with LB_STEREO_MULTIVIEW=1 at launch (read once) and a Vulkan device with
 * multiview, both eyes come from ONE submission of each draw:
 *   - render targets and depth-stencil surfaces created with the back buffer's exact size (not the back buffer
 *     itself; not multisampled, cube, volume or array resources) get a second image layer, and so do targets
 *     the runtime names later (dxvk_stereo_make_layered);
 *   - a pass renders with view mask 0b11 (view i into layer i) when every bound attachment has two layers;
 *     otherwise it is an ordinary mono pass that renders layer 0, as before;
 *   - the vertex shader subtracts the offset of its own view: view 0's in mono passes.
 * Layer 0 is what D3D9 sees everywhere (sampling, StretchRect, GetRenderTargetData, Present); layer 1 leaves
 * the GPU only through dxvk_stereo_get_render_target_data_layer for now. It implies the shader code of
 * LB_STEREO_EYE_UNITS. Unset, nothing changes.
 *
 * LB_STEREO_DUP=<k> (timing experiment, twcclegg/skyrim-apple#23 stage 0): every draw k applies to is issued
 * twice on the DXVK command thread, the copy with the negated offset, into the same render targets. k is the
 * clip-space x shift used while no offset is set here (so =1 works in mono play). The swap chain's
 * "dxvk: gpu busy" log line then ends in ", dup <N> draws per frame". Details in src/dxso/dxso_util.h.
 * Mono only: ignored with LB_STEREO_MULTIVIEW=1, which already draws both eyes in one submission.
 */
#ifndef DXVK_STEREO_H
#define DXVK_STEREO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Clip-space eye shift k for the draws that follow (0 = centre eye). Thread-safe; takes effect at the next draw.
 * The runtime sets it between frames (after Present) so a whole frame renders from one eye.
 * With multiview it sets view 0 to k and view 1 to -k (see set_clip_offset). */
void dxvk_stereo_set_clip_shift(float k);
/* General form: the draws that follow output pos - (x, y, z, w) instead of pos (perspective draws only, as
 * above). Any eye translation t in view space is such a constant: with a perspective P (row vectors),
 * dclip = t * P, e.g. (P00*tx, P11*ty, P22*tz, tz) for a symmetric frustum. Used for VR180 captures, where the
 * eye's offset lies along a turned face's depth axis. set_clip_shift(k) is set_clip_offset(k, 0, 0, 0).
 * Multiview: view 0 (layer 0) gets the offset, view 1 (layer 1) the mirrored -offset, i.e. two eyes either side
 * of a centre camera. Pass the left eye's offset (k = P00*e, e < 0) and layer 0 is the left eye. Mono passes see
 * view 0 only, exactly as before multiview. */
void dxvk_stereo_set_clip_offset(float x, float y, float z, float w);
/* Both views' offsets explicitly (each xyzw as in set_clip_offset; view 1 is used only in multiview passes).
 * Mono passes use view0. Thread-safe; takes effect at the next draw. */
void dxvk_stereo_set_view_clip_offsets(const float view0[4], const float view1[4]);
/* Multiview only: a clip-space matrix per view, column-major, applied before the offsets: clip' = M * clip -
 * offset, for perspective vertices in the main scene's passes (as the offsets). Identity by default. Any other
 * camera for the same scene is exactly such a matrix: M = P_eye * V_eye_from_game * inverse(P_game) (column-vector
 * form), so a view can render a moved, turned or differently framed eye. Thread-safe; takes effect at the next draw. */
void dxvk_stereo_set_view_clip_matrices(const float view0[16], const float view1[16]);
/* Multiview only: per-view clip offsets for w == 1 vertices (pre-transformed and orthographic draws), which the
 * other offsets leave alone. Meant for the interface: set while it is drawn, so it sits at a depth instead of at
 * infinity (equal positions in two parallel eyes), and zero otherwise (full-screen post quads must not move). */
void dxvk_stereo_set_view_ui_offsets(const float view0[4], const float view1[4]);
/* Multiview only: gives the image behind `surface` (IDirect3DSurface9*: level 0 of a one-level render-target or
 * depth-stencil texture, or a standalone depth-stencil surface; D3DPOOL_DEFAULT, not multisampled; standalone
 * render-target surfaces are refused, as nothing could read their second layer) a second layer, as if it had been
 * created as a stereo target: passes whose attachments all have two layers then draw both views, and pixel shaders
 * sample it per view. For a target the size rule above leaves mono that the eyes need their own copy of: the game's
 * deferred sun-shadow mask, smaller than the back buffer below iShadowMaskQuarter=4, whose screen-space result the
 * main pass reads by pixel position.
 *   - The old contents are lost (the new image starts cleared to zero). Call it between frames, or within a frame
 *     only before the target is first drawn into, cleared or sampled in that frame.
 *   - For a colour target and its depth-stencil, layer the depth-stencil first and the colour target only if that
 *     succeeded: a two-layer colour target with a one-layer depth-stencil is drawn in a mono pass, so its layer 1
 *     stays blank while per-view readers sample it. Layering cannot be undone.
 *   - Copies stay single-layer: StretchRect, UpdateSurface, GetRenderTargetData and LockRect read and write layer 0
 *     only, so a two-layer copy destination keeps its old layer 1 (StretchRect and UpdateSurface log this once)
 *     and a copy from a two-layer source holds the first view only.
 * Returns an HRESULT: D3D_OK, also when it already has two layers; D3DERR_INVALIDCALL for a surface it does not
 * take; D3DERR_NOTAVAILABLE if the two-layer image cannot be made; D3DERR_OUTOFVIDEOMEMORY if memory tracking has
 * no room (in these cases the target is unchanged); E_FAIL on any other error. */
long dxvk_stereo_make_layered(void* device, void* surface);
/* The depth of a depth-stencil surface (IDirect3DSurface9*, a 24-bit or float depth format) copied into an R32F
 * render target of the same size (every layer both have), on the GPU: replaces a pass that redraws the scene only
 * to get its depth into a sampleable target. Returns an HRESULT. */
long dxvk_copy_depth_to_float(void* device, void* depthStencil, void* floatTarget);
/* 1 when LB_STEREO_EYE_UNITS or LB_STEREO_MULTIVIEW turned the shader code on, else 0 (and set_clip_shift does
 * nothing visible). */
int dxvk_stereo_enabled(void);
/* 1 when `device` (an IDirect3DDevice9*) renders stereo targets with multiview (LB_STEREO_MULTIVIEW=1 and the
 * Vulkan device supports it), else 0. */
int dxvk_stereo_multiview_enabled(void* device);
/* Image layers behind an IDirect3DSurface9* (2 for a multiview stereo target, else 1; 0 if it has no image). */
int dxvk_stereo_surface_layers(void* surface);
/* Debug and tests: GetRenderTargetData(renderTarget, destSurface) from image layer `layer` of the render target
 * (device: IDirect3DDevice9*, surfaces: IDirect3DSurface9*, destSurface in D3DPOOL_SYSTEMMEM). Layer 0 is plain
 * GetRenderTargetData. Returns an HRESULT: D3DERR_INVALIDCALL for a layer the image does not have. */
long dxvk_stereo_get_render_target_data_layer(void* device, void* renderTarget, void* destSurface, unsigned layer);

#ifdef __cplusplus
}
#endif

#endif /* DXVK_STEREO_H */
