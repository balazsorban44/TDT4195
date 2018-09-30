#version 430 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 in_color;
layout(location = 2) out vec4 out_color;

uniform mat4 cameraMatrix;

uniform mat4 rotationMatrix;

void main()
{

    gl_Position = cameraMatrix * rotationMatrix * position ;

    out_color = in_color;
}

