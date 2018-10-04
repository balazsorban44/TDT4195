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
#include "toolbox.hpp"


double x = 0.0;
double y = 0.0;


auto translation = glm::vec3(1.0f, -5.0f, -20.0f);
float rotate[] = {0.0,0.0,0.0};

void scroll_callback(GLFWwindow* _window, double _xoffset, double yoffset)
{
    translation[2] += yoffset;
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
            translation[0]+=0.5f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            translation[0]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            translation[2]+=0.5f;
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
            translation[2]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            translation[1]-=0.5f;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            translation[1]+=0.5f;
    }
}




void visitSceneNode
        (SceneNode* node, glm::mat4 transformationThusFar, int sceneID, std::stack<glm::mat4>* matrixStack) {

    // Do transformations here
    pushMatrix(matrixStack, transformationThusFar);

    int vaoID = node->vertexArrayObjectID;
    glm::mat4 currTransMat = node->currentTransformationMatrix;
    float refX = node->referencePoint.x;
    float refY = node->referencePoint.y;
    float refZ = node->referencePoint.z;

    currTransMat=
            glm::translate(glm::vec3(refX, refY, refZ))
            * glm::rotate(peekMatrix(matrixStack),float(sin(node->rotation.x)), glm::vec3(node->rotation.x,node->rotation.y,node->rotation.z))
            *  glm::translate(glm::vec3(-refX, -refY, -refZ));

    std::cout << node->rotation.x << std::endl;

    // Do rendering here

    int location = glGetUniformLocation(sceneID, "rotationMatrix");

    glBindVertexArray(vaoID);

    glUniformMatrix4fv(location, 1, GL_FALSE, &currTransMat[0][0]);

    glDrawElements(GL_TRIANGLES, node->VAOIndexCount, GL_UNSIGNED_INT, nullptr);

    for
            (SceneNode* child : node->children) {
        pushMatrix(matrixStack,currTransMat);
        visitSceneNode(child,node->currentTransformationMatrix,sceneID,matrixStack);
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
 //   headNode->referencePoint = float3(0,0,0);
    headNode->rotation.y = 1.0;
   // headNode->position = float3(0,0,0);

    SceneNode* torsoNode = createSceneNode();
    torsoNode->vertexArrayObjectID = torso.getID();
    torsoNode->VAOIndexCount = steve.torso.indices.size();
    torsoNode->referencePoint  = float3(4,0,2);
 //   torsoNode->referencePoint = float3(0,0,0);
    torsoNode->rotation.y = 1.0;
  //  torsoNode->position = float3(0,1,0);

    SceneNode* leftArmNode = createSceneNode();
    leftArmNode->vertexArrayObjectID = leftArm.getID();
    leftArmNode->VAOIndexCount = steve.leftArm.indices.size();
    leftArmNode->referencePoint = float3(4,21,2);
    leftArmNode->rotation.x = -1 * sin(0.1);
 //   leftArmNode->position = float3(1,0,0);

    SceneNode* rightArmNode = createSceneNode();
    rightArmNode->vertexArrayObjectID = rightArm.getID();
    rightArmNode->VAOIndexCount = steve.rightArm.indices.size();
    rightArmNode->referencePoint = float3(12,21,2);
    rightArmNode->rotation.x = 1 * sin(0.1);
  //  rightArmNode->position = float3(1,0,0);

    SceneNode* leftLegNode = createSceneNode();
    leftLegNode->vertexArrayObjectID = leftLeg.getID();
    leftLegNode->VAOIndexCount = steve.leftLeg.indices.size();
    leftLegNode->referencePoint = float3(6,12,0);
    leftLegNode->rotation.x = 1 *0.5 * sin(0.1);
   // leftLegNode->position = float3(1,0,0);

    SceneNode* rightLegNode = createSceneNode();
    rightLegNode->vertexArrayObjectID = rightLeg.getID();
    rightLegNode->VAOIndexCount = steve.rightLeg.indices.size();
    rightLegNode->referencePoint = float3(2,12,0);
    rightLegNode->rotation.x = -1 *0.5 * sin(0.1);
  //  rightLegNode->position = float3(1,0,0);

    SceneNode* chessBoardNode = createSceneNode();
    chessBoardNode->vertexArrayObjectID = chessboard.getID();
    chessBoardNode->VAOIndexCount = terrain.indices.size();
    chessBoardNode->referencePoint = float3(0,0,0);
    chessBoardNode->rotation = float3(0,0,1);

    SceneNode* rootNode = createSceneNode();

    addChild(rootNode,torsoNode);
    addChild(rootNode,chessBoardNode);

    addChild(torsoNode,headNode);
    addChild(torsoNode,leftArmNode);
    addChild(torsoNode,rightArmNode);
    addChild(torsoNode,leftLegNode);
    addChild(torsoNode,rightLegNode);
   // SceneNode* rootNode = createSceneGraph(terrain,steve);


    Path path("../gloom/paths/coordinates_1.txt");


    // Activate shader
    shader.activate();

    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;

    int location = glGetUniformLocation(shader.get(), "cameraMatrix");

    double increment = 0;
    double angle;
    float2 direction = float2();
    float2 currentPosition = float2(torsoNode->position.x,torsoNode->position.z);
    float torsoAngle = 0;


    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {

        model = glm::translate(translation);
        view =
             glm::rotate(model, rotate[0], glm::vec3(1, 0, 0)) *
             glm::rotate(model, rotate[1], glm::vec3(0, 1, 0));
        projection = glm::perspective(3.14159265359f/2, float(windowHeight)/float(windowWidth), 1.0f, 200.0f);

        glm::mat4 mvp = projection * view * model;


        glUniformMatrix4fv(location, 1, GL_FALSE, &mvp[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        increment += getTimeDeltaSeconds();

/*        if (path.hasWaypointBeenReached(currentPosition, 16.0f)){
            path.advanceToNextWaypoint();
        }
        else {
            angle = atan2((currentPosition.y - path.getCurrentWaypoint(16.0f).y), (currentPosition.x - path.getCurrentWaypoint(16.0f)).x);
          //  std::cout << direction.x << "  " << direction.y << std::endl;
            direction.x = (path.getCurrentWaypoint(16.0f).x - currentPosition.x);
            direction.y = (path.getCurrentWaypoint(16.0f).y - currentPosition.y);
            currentPosition.x += currentPosition.x * direction.x;
            currentPosition.y += currentPosition.y * direction.y;
            currentPosition.x = torsoNode->position.x;
            currentPosition.y = torsoNode->position.z;

        }*/

        std::stack<glm::mat4>* matrixStack = createEmptyMatrixStack();



       // torsoAngle = float(sin(increment));

       // std::cout << rightLegNode->rotation.x << std::endl;

        leftArmNode->rotation.x = float(-increment);
        rightArmNode->rotation.x = float(increment);
        leftLegNode->rotation.x = float(0.5*increment);
        rightLegNode->rotation.x = float(-0.5*increment);

        visitSceneNode(rootNode, rootNode->currentTransformationMatrix, shader.get(), matrixStack);




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
