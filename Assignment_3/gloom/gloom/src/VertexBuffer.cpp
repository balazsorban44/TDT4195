//
// Created by balazs on 9/9/18.
//

#include <glad/glad.h>
#include "VertexBuffer.h"
#include "floats.hpp"
#include <vector>

VertexBuffer::VertexBuffer(const void *data, unsigned long length, int index, int attributeSize) {
    glGenBuffers(1, &m_rendererID);
    this->Bind();
    glEnableVertexAttribArray(index);
    glVertexAttribPointer(index, attributeSize,GL_FLOAT,GL_FALSE, sizeof(float4), nullptr);
    glBufferData(GL_ARRAY_BUFFER, length * sizeof(float4), data, GL_STATIC_DRAW);
}


VertexBuffer::~VertexBuffer() {
    glDeleteBuffers(1, &m_rendererID);
}

void VertexBuffer::Bind() const {
    glBindBuffer(GL_ARRAY_BUFFER, m_rendererID);
}

void VertexBuffer::Unbind() const {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

