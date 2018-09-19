#version 430 core

layout(location = 0) out vec4 out_color;
layout(location = 2) in vec4 in_color;


void main()
{
   out_color = in_color;
}
