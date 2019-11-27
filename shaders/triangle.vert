#version 410

uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
uniform float time;


void main()
{
    // Assign varying variables
    v_texcoord  = a_texcoord;
    v_texcoord[0] = v_texcoord[0] * .5*(1 + sin(time));
    v_texcoord[1] = v_texcoord[1] * 1.5*(1 + cos(time));
    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.);

}