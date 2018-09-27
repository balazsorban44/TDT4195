//
// Created by balazs on 9/10/18.
//

#ifndef GLOOM_VERTEXARRAYOBJECT_H
#define GLOOM_VERTEXARRAYOBJECT_H

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "mesh.hpp"

class VertexArrayObject {
private:
    unsigned int m_rendererID;
public:
    explicit VertexArrayObject(Mesh mesh);
    ~VertexArrayObject();

    void Bind() const;
    void Unbind() const;
};


#endif //GLOOM_VERTEXARRAYOBJECT_H
