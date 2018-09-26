//
// Created by balazs on 9/9/18.
//

#ifndef GLOOM_VERTEXBUFFER_H
#define GLOOM_VERTEXBUFFER_H

#include <vector>
#include "floats.hpp"

class VertexBuffer {
private:
    unsigned int m_rendererID;
public:
    VertexBuffer(const void *data, unsigned long size);
    ~VertexBuffer();

    void Bind() const;
    void Unbind() const;
    unsigned int getID() const { return m_rendererID; };

};
#endif //GLOOM_VERTEXBUFFER_H
