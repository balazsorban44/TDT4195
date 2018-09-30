//
// Created by balazs on 9/10/18.
//

#include <glad/glad.h>
#include "VertexArrayObject.h"

VertexArrayObject::VertexArrayObject(Mesh mesh) {
    glGenVertexArrays(1, &m_rendererID);
    this->Bind();
    new VertexBuffer(mesh.vertices.data(), mesh.vertices.size(), 0, 4);
    new VertexBuffer(mesh.colours.data(), mesh.colours.size(), 1, 4);
    new IndexBuffer(mesh.indices.data(), mesh.indices.size());
}


VertexArrayObject::~VertexArrayObject() {
    glDeleteBuffers(1, &m_rendererID);
}


void VertexArrayObject::Bind() const {
    glBindVertexArray(m_rendererID);
}

void VertexArrayObject::Unbind() const {
    glBindVertexArray(0);
}

