//
// Created by balazs on 9/9/18.
//

#ifndef GLOOM_VERTEXBUFFER_H
#define GLOOM_VERTEXBUFFER_H
class VertexBuffer {
private:
    unsigned int m_rendererID;
public:
    VertexBuffer(const void* data, unsigned int size);
    ~VertexBuffer();

    void Bind() const;
    void Unbind() const;
    unsigned int getID() const { return m_rendererID; };

};
#endif //GLOOM_VERTEXBUFFER_H
