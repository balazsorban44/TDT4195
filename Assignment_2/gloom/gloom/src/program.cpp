// Local headers
#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"
#include "iostream"
#include "math.h"

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArrayObject.h"

//  Coordinates DATA



float coords[] = {
       -0.6, -0.6, 0,
        0.6, -0.6, 0,
        0,    0.6, 0,
};

float colors[] = {
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0
};

unsigned int indices[] = {0,1,2};




void runProgram(GLFWwindow* window)
{
    Gloom::Shader shader;
    shader.makeBasicShader("../gloom/shaders/simple.vert",
                       "../gloom/shaders/simple.frag");

    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Configure miscellaneous OpenGL settings
    glEnable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up your scene here (create Vertex Array Objects, etc.)
    VertexArrayObject vao;

    VertexBuffer vb(coords, 3 * 3 * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, sizeof(float) * 3, nullptr);

    VertexBuffer vb2(colors, 3 * 4 * sizeof(float));
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE, sizeof(float) * 4, nullptr);

    IndexBuffer ib(indices, 3);

    // Activate shader
    shader.activate();
    ib.Bind();

    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {




        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);

        

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
}
