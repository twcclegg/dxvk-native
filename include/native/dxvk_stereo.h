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
 *     itself; not multisampled, cube, volume or array resources) get a second image layer;
 *   - a pass renders with view mask 0b11 (view i into layer i) when every bound attachment has two layers;
 *     otherwise it is an ordinary mono pass that renders layer 0, as before;
 *   - the vertex shader subtracts the offset of its own view: view 0's in mono passes.
 * Layer 0 is what D3D9 sees everywhere (sampling, StretchRect, GetRenderTargetData, Present); layer 1 leaves
 * the GPU only through dxvk_stereo_get_render_target_data_layer for now. It implies the shader code of
 * LB_STEREO_EYE_UNITS. Unset, nothing changes.
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
