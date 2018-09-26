//
// Created by balazs on 9/9/18.
//

#include <glad/glad.h>
#include "IndexBuffer.h"


IndexBuffer::IndexBuffer(const void *data, unsigned long count)
    : m_Count(count)
{
    glGenBuffers(1, &m_rendererID);
    this->Bind();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned long), data, GL_STATIC_DRAW);
}

IndexBuffer::~IndexBuffer() {
    glDeleteBuffers(1, &m_rendererID);
}

void IndexBuffer::Bind() const {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_rendererID);
}

void IndexBuffer::Unbind() const {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
