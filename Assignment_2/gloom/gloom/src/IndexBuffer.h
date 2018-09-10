//
// Created by balazs on 9/9/18.
//

#ifndef GLOOM_INDEXBUFFER_H
#define GLOOM_INDEXBUFFER_H
class IndexBuffer {
private:
    unsigned int m_rendererID;
    unsigned int m_Count;
public:
    IndexBuffer(const unsigned int* data, unsigned int count);
    ~IndexBuffer();

    void Bind() const;
    void Unbind() const;

    inline unsigned int GetCount() const { return m_Count;};
};
#endif //GLOOM_INDEXBUFFER_H
