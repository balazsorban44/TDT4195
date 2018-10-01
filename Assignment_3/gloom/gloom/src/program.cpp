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
#include "sceneGraph.hpp"
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>


double x = 0.0;
double y = 0.0;


auto model = glm::vec3(1.0f, -5.0f, -20.0f);
float rotate[] = {0.0,0.0,0.0};

void scroll_callback(GLFWwindow* _window, double _xoffset, double yoffset)
{
    model[2] += yoffset;
}

void handleInput(GLFWwindow* window)
{
    // Mouse inputs
    double xCurr, yCurr;
    glfwGetCursorPos(window, &xCurr, &yCurr);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        rotate[1] += static_cast<float>(xCurr - x)/10000;
        rotate[0] += static_cast<float>(yCurr - y)/10000;
    } else {
        x = xCurr;
        y = yCurr;
    }
    glfwSetScrollCallback(window, scroll_callback);


    //Keyboard inputs
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            rotate[1] += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            rotate[1] -= 0.01f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            rotate[0] += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            rotate[0] -= 0.01f;
    }
    else {
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            model[0]+=0.5f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            model[0]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            model[2]+=0.5f;
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
            model[2]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            model[1]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            model[1]+=0.5f;
    }
}




SceneNode* createSceneGraph(Mesh terrain, MinecraftCharacter steve){

    VertexArrayObject head(steve.head);
    VertexArrayObject torso(steve.torso);
    VertexArrayObject leftArm(steve.leftArm);
    VertexArrayObject rightArm(steve.rightArm);
    VertexArrayObject leftLeg(steve.leftLeg);
    VertexArrayObject rightLeg(steve.rightLeg);

    VertexArrayObject chessboard(terrain);

    // Nodes
    SceneNode* headNode = createSceneNode();
    headNode->vertexArrayObjectID = head.getID();
    headNode->VAOIndexCount = steve.head.indices.size();
    headNode->referencePoint = float3(8,28,4);

    SceneNode* torsoNode = createSceneNode();
    torsoNode->vertexArrayObjectID = torso.getID();
    torsoNode->VAOIndexCount = steve.torso.indices.size();
    torsoNode->referencePoint  = float3(4,0,2);

    SceneNode* leftArmNode = createSceneNode();
    leftArmNode->vertexArrayObjectID = leftArm.getID();
    leftArmNode->VAOIndexCount = steve.leftArm.indices.size();
    leftArmNode->referencePoint = float3(4,21,2);

    SceneNode* rightArmNode = createSceneNode();
    rightArmNode->vertexArrayObjectID = rightArm.getID();
    rightArmNode->VAOIndexCount = steve.rightArm.indices.size();
    rightArmNode->referencePoint = float3(12,21,2);

    SceneNode* leftLegNode = createSceneNode();
    leftLegNode->vertexArrayObjectID = leftLeg.getID();
    leftLegNode->VAOIndexCount = steve.leftLeg.indices.size();
    leftLegNode->referencePoint = float3(6,12,0);

    SceneNode* rightLegNode = createSceneNode();
    rightLegNode->vertexArrayObjectID = rightLeg.getID();
    rightLegNode->VAOIndexCount = steve.rightLeg.indices.size();
    rightLegNode->referencePoint = float3(2,12,0);

    SceneNode* chessBoardNode = createSceneNode();
    chessBoardNode->vertexArrayObjectID = chessboard.getID();
    chessBoardNode->VAOIndexCount = terrain.indices.size();
    chessBoardNode->referencePoint = float3(0,0,0);

    SceneNode* rootNode = createSceneNode();

    addChild(rootNode, torsoNode);
    addChild(rootNode, chessBoardNode);

    addChild(torsoNode, headNode);
    addChild(torsoNode, leftArmNode);
    addChild(torsoNode, rightArmNode);
    addChild(torsoNode, leftLegNode);
    addChild(torsoNode, rightLegNode);

    return rootNode;
}


void visitSceneNode(
        SceneNode* node,
        glm::mat4 transformationThusFar,
        int sceneID,
        std::stack<glm::mat4>* matrixStack,
        double increment
        ) {

    // Do transformations here
    pushMatrix(matrixStack, transformationThusFar);

    int vaoID = node->vertexArrayObjectID;
    glm::mat4 currTransMat = node->currentTransformationMatrix;
    float refX = node->referencePoint.x;
    float refY = node->referencePoint.y;
    float refZ = node->referencePoint.z;

    // Torso & Head
    if (vaoID == 2 || vaoID == 1)
        currTransMat = glm::translate(glm::vec3(0, 0, increment*5));

    // Arms
    if (vaoID == 4 || vaoID == 3) {
        int direction = vaoID == 3 ? -1 : 1;
        currTransMat =
                glm::translate(glm::vec3(refX, refY, refZ)) *
                glm::rotate(peekMatrix(matrixStack), float(direction * sin(increment)), glm::vec3(1,0,0)) *
                glm::translate(glm::vec3(-refX, -refY, -refZ));
    }

    // Legs
    if (vaoID == 5 || vaoID == 6) {
        int direction = vaoID == 6 ? -1 : 1;
        currTransMat =
                glm::translate(glm::vec3(refX, refY, refZ)) *
                glm::rotate(peekMatrix(matrixStack), float(direction* 0.5 * sin(increment)), glm::vec3(1,0,0)) *
                glm::translate(glm::vec3(-refX, -refY, -refZ));
    }

    // Do rendering here

    int location = glGetUniformLocation(sceneID, "rotationMatrix");

    glBindVertexArray(vaoID);

    glUniformMatrix4fv(location, 1, GL_FALSE, &currTransMat[0][0]);

    glDrawElements(GL_TRIANGLES, node->VAOIndexCount, GL_UNSIGNED_INT, nullptr);

    for (SceneNode* child : node->children)
    {
        pushMatrix(matrixStack,currTransMat);
        visitSceneNode(child, currTransMat, sceneID, matrixStack, increment);
        popMatrix(matrixStack);
    }

    popMatrix(matrixStack);
}


void runProgram(GLFWwindow* window)
{
    Gloom::Shader shader;
    shader.makeBasicShader(
            "../gloom/shaders/simple.vert",
            "../gloom/shaders/simple.frag"
            );

    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up your scene here (create Vertex Array Objects, etc.)

    MinecraftCharacter steve = loadMinecraftCharacterModel("../gloom/res/steve.obj");

    Mesh terrain = generateChessboard(7, 5, 16.0f, float4(1, 0.603, 0, 1.0), float4(0.172, 0.172, 0.172, 1.0));

    SceneNode* rootNode = createSceneGraph(terrain, steve);


    // Activate shader
    shader.activate();

    glm::mat4 projection;
    glm::mat4 view;

    int location = glGetUniformLocation(shader.get(), "cameraMatrix");

    double increment = 0;


    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {

        projection = glm::perspective(3.14159265359f/2, float(windowHeight)/float(windowWidth), 1.0f, 200.0f);
        view =
             glm::rotate(glm::translate(model), rotate[0], glm::vec3(1, 0, 0)) *
             glm::rotate(glm::translate(model), rotate[1], glm::vec3(0, 1, 0));

        glm::mat4 mvp_matrix = projection * view * glm::translate(model);


        glUniformMatrix4fv(location, 1, GL_FALSE, &mvp_matrix[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        increment += getTimeDeltaSeconds();
        std::stack<glm::mat4>* matrixStack = createEmptyMatrixStack();
        visitSceneNode(rootNode, glm::mat4(), shader.get(), matrixStack, increment);


        // Handle other events
        glfwPollEvents();
        handleInput(window);

        // Flip buffers
        glfwSwapBuffers(window);

    }
    // Deactivate / destroy shader
    shader.deactivate();
    shader.destroy();
}
