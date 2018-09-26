// Local headers
#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"
#include "iostream"
#include "math.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArrayObject.h"
#include "OBJLoader.hpp"
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>




unsigned int indices[] = {6,7,8,3,4,5,0,1,2};

glm::vec3 currentMotion = glm::vec3(1.0f,1.0f,-1.0f); //c) a)

float axisRotation[] = {0.0,0.0,0.0};  // x, y, z



void runProgram(GLFWwindow* window)
{
    Gloom::Shader shader;
    shader.makeBasicShader("../gloom/shaders/simple.vert",
                       "../gloom/shaders/simple.frag");

    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up your scene here (create Vertex Array Objects, etc.)


    MinecraftCharacter steve = loadMinecraftCharacterModel("../gloom/res/steve.obj");


    VertexArrayObject head;




    VertexBuffer headCoords(steve.head.vertices.data(), steve.head.vertices.size());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE, sizeof(float) * 4, nullptr);

    VertexBuffer headColors(steve.head.colours.data(), steve.head.colours.size());
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE, sizeof(float) * 4, nullptr);

    IndexBuffer ib(steve.head.indices.data(), steve.head.indices.size());


    // Activate shader
    shader.activate();
    ib.Bind();

    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    glm::mat4 rotateX;
    glm::mat4 rotateY;
    int location = glGetUniformLocation(shader.get(), "cameraMatrix");


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


        //glUniformMatrix4fv(location, 1,GL_FALSE,&mvp_matrix[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, steve.head.indices.size(), GL_UNSIGNED_INT, nullptr);

        // Handle other events
        glfwPollEvents();
        handleKeyboardInput(window);
        

        // Flip buffers
        glfwSwapBuffers(window);

    }
    // Deactivate / destroy shader
    shader.deactivate();
    shader.destroy();
}


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
        if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
        {
            currentMotion[2]+=0.01f;
        }
        // Use bottom arrow to go backward (translation)
        if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
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