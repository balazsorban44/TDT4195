//
// Created by balazs on 9/10/18.
//

#ifndef GLOOM_VERTEXARRAYOBJECT_H
#define GLOOM_VERTEXARRAYOBJECT_H

class VertexArrayObject {
private:
    unsigned int m_rendererID;
public:
    VertexArrayObject();
    ~VertexArrayObject();

    void Bind() const;
    void Unbind() const;
};


#endif //GLOOM_VERTEXARRAYOBJECT_H
