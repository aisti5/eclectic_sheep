#version 410

uniform sampler2D u_texture_from;// Texture 1
uniform sampler2D u_texture_to;// Texture 2
varying vec2      v_texcoord;// Interpolated fragment texture coordinates (in)
//out  vec4 outColor;

uniform float time;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
    // Get texture color

    vec4 t_color_from = texture2D(u_texture_from, v_texcoord);
    vec4 t_color_to = texture2D(u_texture_to, v_texcoord);
    // Final color
//    for (int i=0; i < 1000; i++){
//        gl_FragColor = (1-i/1000)*t_color_from + (i/1000)*t_color_to;
//    }

    vec4 t_color_sin = vec4(t_color_from[0]*sin(time), t_color_from[1]*cos(time), t_color_from[2], 1);
    t_color_sin = mix(t_color_from, t_color_to, (1 + sin(time))/2);
    gl_FragColor = t_color_sin;//t_color_from; //mix(t_color_from, t_color_to, step(0.4, v_texcoord.x));


}