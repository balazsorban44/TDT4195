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
#include "toolbox.hpp"
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>






glm::vec3 currentMotion = glm::vec3(1.0f,1.0f,-20.0f); //c) a)

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

   /* float4 coords2[] = steve.head.vertices.data();
    float4 colors2[] = steve.head.colours.data();*/


    // Set up your scene here (create Vertex Array Objects, etc.)
    VertexArrayObject head(steve.head);
    VertexArrayObject torso(steve.torso);
    VertexArrayObject leftArm(steve.leftArm);
    VertexArrayObject rightArm(steve.rightArm);
    VertexArrayObject leftLeg(steve.leftLeg);
    VertexArrayObject rightLeg(steve.rightLeg);
    
    Mesh chessboardMesh = generateChessboard(7, 5, 16.0f, float4(1, 0.603, 0, 1.0), float4(0.172, 0.172, 0.172, 1.0));
    VertexArrayObject chessboard(chessboardMesh);


    // Activate shader
    shader.activate();

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


        glUniformMatrix4fv(location, 1,GL_FALSE,&mvp_matrix[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



        chessboard.Bind();
        glDrawElements(GL_TRIANGLES, chessboardMesh.vertices.size(), GL_UNSIGNED_INT, nullptr);
        head.Bind();
        glDrawElements(GL_TRIANGLES, steve.head.vertices.size(), GL_UNSIGNED_INT, nullptr);
        torso.Bind();
        glDrawElements(GL_TRIANGLES, steve.torso.vertices.size(), GL_UNSIGNED_INT, nullptr);
        leftArm.Bind();
        glDrawElements(GL_TRIANGLES, steve.leftArm.vertices.size(), GL_UNSIGNED_INT, nullptr);

        rightArm.Bind();
        glDrawElements(GL_TRIANGLES, steve.rightArm.vertices.size(), GL_UNSIGNED_INT, nullptr);

        leftLeg.Bind();
        glDrawElements(GL_TRIANGLES, steve.leftLeg.vertices.size(), GL_UNSIGNED_INT, nullptr);
        rightLeg.Bind();
        glDrawElements(GL_TRIANGLES, steve.rightLeg.vertices.size(), GL_UNSIGNED_INT, nullptr);
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
            currentMotion[0]+=0.5f;
        }

        // Use right arrow to move to the right the camera (translation)
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            currentMotion[0]-=0.5f;
        }
        // Use up arrow to go forward (translation)
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        {
            currentMotion[2]+=0.5f;
        }
        // Use bottom arrow to go backward (translation)
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        {
            currentMotion[2]-=0.5f;
        }

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            currentMotion[1]-=0.5f;
        }

        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            currentMotion[1]+=0.5f;
        }


    }


}