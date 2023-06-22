// Container for graphics objects header

#ifndef _AURORA_GRAPHICSENTITY_HEADER_
#define _AURORA_GRAPHICSENTITY_HEADER_

#include <gl/glew.h>
#include <glm.hpp>
#include <vector>

class PhysicsEntity;

class GraphicsEntity {
public:

	GraphicsEntity() {
		glGenBuffers(1, &m_VBO);
		glGenBuffers(1, &m_EBO);
	};

	void bind();

	// Vertex and Element Buffers
	GLuint m_VBO;
	GLuint m_EBO;

	// Graphics properties
	std::vector<glm::vec3>* m_position;
	std::vector<glm::uvec3>* m_indices;
	std::vector<glm::vec2>* m_uv;
	std::vector<glm::vec3>* m_normal;
	std::vector<glm::vec3>* m_color;

	// Model matrix for transformations
	glm::mat4 m_model_matrix;

	PhysicsEntity* m_mirror_pe;

	// 
	bool m_has_material{ false };
	bool m_has_normal{ false };
	bool m_has_texture{ false };

	GLuint m_attrib_num;

private:

};

#endif // _AURORA_GRAPHICSENTITY_HEADER_