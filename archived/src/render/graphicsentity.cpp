// Container for graphics objects exec

#include "graphicsentity.hpp"

void GraphicsEntity::bind() {
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
}