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
 * LB_STEREO_DUP=<k> (timing experiment, twcclegg/skyrim-apple#23 stage 0): every draw k applies to is issued
 * twice on the DXVK command thread, the copy with the negated offset, into the same render targets. k is the
 * clip-space x shift used while no offset is set here (so =1 works in mono play). The swap chain's
 * "dxvk: gpu busy" log line then ends in ", dup <N> draws per frame". Details in src/dxso/dxso_util.h.
 */
#ifndef DXVK_STEREO_H
#define DXVK_STEREO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Clip-space eye shift k for the draws that follow (0 = centre eye). Thread-safe; takes effect at the next draw.
 * The runtime sets it between frames (after Present) so a whole frame renders from one eye. */
void dxvk_stereo_set_clip_shift(float k);
/* General form: the draws that follow output pos - (x, y, z, w) instead of pos (perspective draws only, as
 * above). Any eye translation t in view space is such a constant: with a perspective P (row vectors),
 * dclip = t * P, e.g. (P00*tx, P11*ty, P22*tz, tz) for a symmetric frustum. Used for VR180 captures, where the
 * eye's offset lies along a turned face's depth axis. set_clip_shift(k) is set_clip_offset(k, 0, 0, 0). */
void dxvk_stereo_set_clip_offset(float x, float y, float z, float w);
/* 1 when LB_STEREO_EYE_UNITS turned the shader code on, else 0 (and set_clip_shift does nothing visible). */
int dxvk_stereo_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* DXVK_STEREO_H */
