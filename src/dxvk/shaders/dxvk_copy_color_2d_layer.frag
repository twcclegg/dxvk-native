#version 450

layout(set = 0, binding = 0)
uniform sampler2DArray s_image;

layout(location = 0) out vec4 o_color;

// The layer comes from the push constants (one draw per layer) instead of
// gl_Layer, which a fragment shader can only read with the Geometry capability.
layout(push_constant)
uniform u_info_t {
  ivec2 offset;
  int   layer;
} u_info;

void main() {
  o_color = texelFetch(s_image,
    ivec3(gl_FragCoord.xy + u_info.offset, u_info.layer), 0);
}
