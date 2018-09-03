// Local headers
#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"



int setUpVAO(float* coords, unsigned int* indexes, int elementCount, int indexCount) {
    unsigned int arrayIDs = 0;
    glGenVertexArrays(1, &arrayIDs);
    glBindVertexArray(arrayIDs);
    glGenBuffers(1, &arrayIDs);
    glBindBuffer(GL_ARRAY_BUFFER, arrayIDs);
    glBufferData(GL_ARRAY_BUFFER, elementCount * sizeof(float), coords, GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,12, 0);
    glEnableVertexAttribArray(0);


    // Generate index buffer
    unsigned int indexIDs = 1;
    glGenBuffers(1, &indexIDs);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexIDs);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indexes, GL_STATIC_DRAW);
    return 0;
};



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


    float coords_1[] = {
        0.25, 0.25, 0,
        1, 0, 0,
        0, 1, 0,

        -0.25, -0.25, 0,
        -1, 0, 0,
        0, -1, 0,
    
        0.25, -0.25, 0,
        0, -1, 0,
        1, 0, 0,

        -0.25, 0.25, 0,
        0, 1, 0,
        -1, 0, 0,

        
        -0.2, -0.2, 0,
        0, -0.2, 0,
        -0.2, 0, 0,
    };

    float coords_2[] = {
        0.6, -0.8, -1.2,
        0, 0.4, 0,
        -0.8, -0.2, 1.2
    };


    unsigned int indexes_1[] = {0,1,2, 3,4,5,6,7,8,9, 10, 11, 12, 13, 14};
    unsigned int indexes_2[] = {0,1,2};
    
    // Set up your scene here (create Vertex Array Objects, etc.)
    setUpVAO(coords_1, indexes_1, 45, 15);
    // setUpVAO(coords_2, indexes_2, 9, 3);

    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.activate();
        // Draw your scene here
        
        // Task 1
        glDrawElements(GL_TRIANGLES, 15, GL_UNSIGNED_INT, 0);

        // Task 2
        // glDrawElements(GL_TRIANGLES, 15, GL_UNSIGNED_INT, 0);
        

        // Handle other events
        glfwPollEvents();
        handleKeyboardInput(window);
        
        // Deactivate shader program
        shader.deactivate();

        // Flip buffers
        glfwSwapBuffers(window);
    }
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
