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


/**
 * Change the path easily by changing the number on line 25.
 * 0, 1, or 2
 */
std::string coordPath = "../gloom/paths/coordinates_1.txt";


double x = 0.0;
double y = 0.0;

/**
 * Width and height of a single chessboard tile
 */
float tileWidth = 16.0f;

glm::vec3 translation(-tileWidth * 2, -tileWidth, -8 * tileWidth);

glm::vec2 rotate;

void scroll_callback(GLFWwindow* _window, double _xoffset, double yoffset)
{
    translation[2] += yoffset*4;
}

void handleInput(GLFWwindow* window)
{
    // Mouse inputs
    double xCurr, yCurr;
    glfwGetCursorPos(window, &xCurr, &yCurr);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        rotate.y += static_cast<float>(xCurr - x)/5000;
        rotate.x += static_cast<float>(yCurr - y)/5000;
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
            rotate.y += 0.05f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            rotate.y -= 0.05f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            rotate.x += 0.05f;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            rotate.x -= 0.05f;
    }
    else {
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            translation.x+=1;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            translation.x-=1;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            translation.z+=1;
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            translation.z-=1;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            translation.y-=1;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            translation.y+=1;
    }
}


auto matrixStack = createEmptyMatrixStack();

void visitSceneNode(SceneNode* node, glm::mat4 transformationThusFar, Gloom::Shader* shader)
{


    int vaoID = node->vertexArrayObjectID;
    auto [refX, refY, refZ] = node->referencePoint;
    auto [posX, posY, posZ] = node->position;
    auto [rotX, rotY, rotZ] = node->rotation;

    /**
     * The model transformation 🗡
     */
    node->currentTransformationMatrix =
         transformationThusFar *
         glm::translate(glm::vec3(posX, posY, posZ)) *
         glm::translate(glm::vec3(refX, refY, refZ)) *
         glm::rotate(glm::mat4(),  rotX, glm::vec3(1, 0, 0)) *
         glm::rotate(glm::mat4(), rotY, glm::vec3(0, 1, 0)) *
         glm::rotate(glm::mat4(), rotZ, glm::vec3(0, 0, 1)) *
         glm::translate(glm::vec3(-refX, -refY, -refZ));


    /**
     * Rendering the node
     * NOTE: Do not need to call OpenGL if there is nothing to render.
     */

    if (vaoID > -1) {
        int location = glGetUniformLocation(shader->get(), "model");
        glBindVertexArray(vaoID);
        glUniformMatrix4fv(location, 1, GL_FALSE, &node->currentTransformationMatrix[0][0]);
        glDrawElements(GL_TRIANGLES, node->VAOIndexCount, GL_UNSIGNED_INT, nullptr);
    }

    /**
     * Visit each child
     */
    for (SceneNode* child : node->children)
        visitSceneNode(child, node->currentTransformationMatrix, shader);

}


void runProgram(GLFWwindow* window)
{
    Gloom::Shader shader;
    shader.makeBasicShader("../gloom/shaders/simple.vert", "../gloom/shaders/simple.frag");

    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up your scene here (create Vertex Array Objects, etc.)

    MinecraftCharacter steve = loadMinecraftCharacterModel("../gloom/res/steve.obj");

    Mesh terrain =
            generateChessboard(7, 5, tileWidth, float4(1, 0.603, 0, 1.0), float4(0.172, 0.172, 0.172, 1.0));

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

    SceneNode* rootNode = createSceneNode();

    addChild(rootNode,torsoNode);
    addChild(rootNode,chessBoardNode);

    addChild(torsoNode,headNode);
    addChild(torsoNode,leftArmNode);
    addChild(torsoNode,rightArmNode);
    addChild(torsoNode,leftLegNode);
    addChild(torsoNode,rightLegNode);


    Path path(coordPath);


    // Activate shader
    shader.activate();

    glm::mat4 projection;
    glm::mat4 view;

    int location = glGetUniformLocation(shader.get(), "projectionView");

    double increment = 0;

    

    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {

        view =
             glm::translate(translation) *
             glm::rotate(glm::mat4(), rotate.x, glm::vec3(1, 0, 0)) *
             glm::rotate(glm::mat4(), rotate.y, glm::vec3(0, 1, 0));


        projection =
                glm::perspective(
                        3.14159265358979323846f/2,
                        float(windowHeight)/float(windowWidth),
                        1.0f,
                        200.0f
                        );

        glm::mat4 pv = projection * view;


        glUniformMatrix4fv(location, 1, GL_FALSE, &pv[0][0]);

        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        /**
         * Get that increment increasing
         */
        increment += getTimeDeltaSeconds();

        /**
         * Swinging the arms and legs
         * NOTE: Legs are swinging less because it looks more natural
         */
        leftArmNode->rotation.x = static_cast<float>(-cos(increment) * 0.75);
        rightArmNode->rotation.x = static_cast<float>(cos(increment) * 0.75);
        leftLegNode->rotation.x = static_cast<float>(cos(increment) * 0.5);
        rightLegNode->rotation.x = static_cast<float>(-cos(increment) * 0.5);

        
        /**
         * Follow the path
         */
         
        if(path.hasWaypointBeenReached(float2(torsoNode->position.x, torsoNode->position.z), tileWidth))
            path.advanceToNextWaypoint();


        glm::vec2 dPosition((path.getCurrentWaypoint(tileWidth).x - torsoNode->position.x),
                            path.getCurrentWaypoint(tileWidth).y - torsoNode->position.z);

        dPosition = glm::normalize(dPosition);

        double dRotation(atan2(dPosition.x, dPosition.y));

        torsoNode->position.x += dPosition.x / 4;
        torsoNode->position.z += dPosition.y / 4;
        torsoNode->rotation.y = dRotation;

        visitSceneNode(rootNode, glm::mat4(), &shader);


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
