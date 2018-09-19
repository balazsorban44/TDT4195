#version 430 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 in_color;
layout(location = 2) out vec4 out_color;

uniform float PointValue;
uniform mat4 projectionMatrix;

mat4 matrix = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
//mat4 matrix = {{a,d,0,0},{b,e,0,0},{0,0,1,0},{c,f,0,1}};
void main()
{
/*   // change of a
    matrix[0][0] = PointValue;
    //change of b
    matrix[1][0] = PointValue;
    //change of c
    matrix[3][0] = PointValue;
    //change of d
    matrix[0][1] = PointValue;
    //change of e
    matrix[1][1] = PointValue;
   //change of f
    matrix[3][1] = PointValue;*/

   // gl_Position = matrix * position;

    gl_Position = projectionMatrix * position;

   // gl_Position = position;

    out_color = in_color;
}

