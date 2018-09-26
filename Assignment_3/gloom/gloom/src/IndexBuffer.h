//
// Created by balazs on 9/9/18.
//

#ifndef GLOOM_INDEXBUFFER_H
#define GLOOM_INDEXBUFFER_H

#include <vector>

class IndexBuffer {
private:
    unsigned int m_rendererID;
    unsigned long m_Count;
public:
    IndexBuffer(const void *data, unsigned long count);
    ~IndexBuffer();

    void Bind() const;
    void Unbind() const;

    inline unsigned long GetCount() const { return m_Count;};
};
#endif //GLOOM_INDEXBUFFER_H
