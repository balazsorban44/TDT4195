//
// Created by balazs on 9/10/18.
//

#include <glad/glad.h>
#include "VertexArrayObject.h"

VertexArrayObject::VertexArrayObject() {
    glGenVertexArrays(1, &m_rendererID);
    glBindVertexArray(m_rendererID);
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

