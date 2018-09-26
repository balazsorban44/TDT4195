// Local headers
#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"
#include "iostream"
#include "math.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArrayObject.h"
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

//using namespace glm

//  Coordinates DATA

//Task 1 b
/*float coords[] = {

        -0.24, 0, 0,
        0.24, 0, 0,
        0, 0.45, 0,

        -0.20, -0.05, 0,
        0, -0.45, 0,
        0.2, -0.05, 0,

        0, -0.5, 0,
        0.5, -0.5, 0,
        0.25,    0, 0,

        -0.5, -0.5, 0,
        0,-0.5,0,
        -0.25,0,0,

        -0.6, -0.6, 0,
        0.6, -0.6, 0,
        0,    0.6, 0,


};

float colors[] = {
        1.0, 0.0, 0.8, 1.0,
        0.8, 1.0, 0.0, 1.0,
        0.0, 0.8, 1.0, 1.0,

        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,

        1.0, 0.5, 0.0, 1.0,
        0.5, 1.0, 0.8, 1.0,
        0.8, 0.4, 1.0, 1.0,

        0.8, 0.5, 0.0, 1.0,
        0.5, 0.0, 0.8, 1.0,
        0.8, 0.6, 1.0, 1.0,

        0.0, 0.8, 0.8, 1.0,
        0.8, 0.8, 0.0, 1.0,
        0.8, 0.0, 0.8, 1.0,

};

unsigned int indices[] = {3,4,5,0,1,2,6,7,8,9,10,11,12,13,14};*/

//Task 2
float coords[] = {

        -0.2, 0.45, 0.8,
        0, 0, 0.8,
        0.65, 0.45, 0.8,

       -0.1, 0, -0.5,
        0.5, 0, -0.5,
        -0.3, 0.45, -0.5,

        0.1, 0, 0.0,
        0.6, 0, 0.0,
        0.35, 0.45, 0.0
};

float colors[] = {

//blue
        0.0, 0.8, 1.0, 0.3,
        0.0, 0.8, 1.0, 0.3,
        0.0, 0.8, 1.0, 0.3,

//yellow
        0.8, 1.0, 0.0, 0.8,
        0.8, 1.0, 0.0, 0.8,
        0.8, 1.0, 0.0, 0.8,
//pink
        1.0, 0.0, 0.8, 0.5,
        1.0, 0.0, 0.8, 0.5,
        1.0, 0.0, 0.8, 0.5


};


unsigned int indices[] = {6,7,8,3,4,5,0,1,2};

glm::vec3 currentMotion = glm::vec3(1.0f,1.0f,-1.0f); //c) a)

float axisRotation[] = {0.0,0.0,0.0};  // x, y, z
glm::mat4 rotationX = glm::mat4({{1,0,0,0},{0,1,0,0},{0,0,0,1},{0,0,0,1}} );
glm::mat4 rotationY = glm::mat4({{1,0,0,0},{0,1,0,0},{0,0,0,1},{0,0,0,1}} );



void runProgram(GLFWwindow* window)
{
    Gloom::Shader shader;
    shader.makeBasicShader("../gloom/shaders/simple.vert",
                       "../gloom/shaders/simple.frag");

    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Configure miscellaneous OpenGL settings
    //To see the back faces of the triangles we commented out the next line
   // glEnable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up your scene here (create Vertex Array Objects, etc.)
    VertexArrayObject vao;

    VertexBuffer vb(coords, 9 * 3 * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, sizeof(float) * 3, nullptr);

    VertexBuffer vb2(colors, 9 * 4 * sizeof(float));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE, sizeof(float) * 4, nullptr);

    IndexBuffer ib(indices, 9);

    // Activate shader
    shader.activate();
    ib.Bind();

    float OsciValue = -0.5;
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    glm::mat4 rotateX;
    glm::mat4 rotateY;


    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {

        projection = glm::perspective(40.0f,float(windowHeight) / float(windowWidth),1.0f,100.0f);
        view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0,0.0,-1.0));
        model = glm::translate(glm::mat4(1.0f), currentMotion);
        rotateX = glm::rotate(model, axisRotation[0],glm::vec3(1,0,0));
        rotateY = glm::rotate(model, axisRotation[1],glm::vec3(0,1,0));

        glm::mat4 mvp_matrix = projection * rotateX * rotateY * view * model;

        shader.activate();

      // int location_1 = glGetUniformLocation(shader.get(), "PointValue");
        int location_2 = glGetUniformLocation(shader.get(), "cameraMatrix");

      //  glUniform1f(location_1, sin(OsciValue));
        glUniformMatrix4fv(location_2, 1,GL_FALSE,&mvp_matrix[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, nullptr);


        // Handle other events
        glfwPollEvents();
        handleKeyboardInput(window);
        

        // Flip buffers
        glfwSwapBuffers(window);

        //slightly incremented each frame (between -0.5 and 0.5)

       // OsciValue+=0.01;


    }
    // Deactivate / destroy shader
    shader.deactivate();
    shader.destroy();
}


/*void handleKeyboardInput(GLFWwindow* window)
{
    // Use escape key for terminating the GLFW window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}*/

void handleKeyboardInput(GLFWwindow* window)
{

    // Use escape key for terminating the GLFW window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if( glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            axisRotation[1]+=0.01f;
        }
        // Use bottom arrow to go backward (translation)
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            axisRotation[1]-=0.01f;
        }
        // Use up arrow to go up (translation)
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            axisRotation[0]+=0.01f;
        }
        // Use bottom arrow to go backward (translation)
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            axisRotation[0]-=0.01f;
        }



    }

    else {

        // Use left arrow to move to the left the camera (translation)
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            currentMotion[0]+=0.01f;
        }

        // Use right arrow to move to the right the camera (translation)
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            currentMotion[0]-=0.01f;
        }
        // Use up arrow to go forward (translation)
        if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS)
        {
            currentMotion[2]+=0.01f;
        }
        // Use bottom arrow to go backward (translation)
        if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
        {
            currentMotion[2]-=0.01f;
        }

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            currentMotion[1]-=0.01f;
        }

        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            currentMotion[1]+=0.01f;
        }


    }


}