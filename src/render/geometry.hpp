// Mesh classes

#ifndef _AURORA_GEOMETRY_HEADER_
#define _AURORA_GEOMETRY_HEADER_

#include <glm.hpp>
#include "renderutils.hpp"

class Mesh {
public:

	virtual void bind() = 0;

protected:

	GLuint _VBO;
	GLuint _EBO;
};

class SphereMesh : public Mesh {
public:

	SphereMesh(float radius, unsigned int stack_count, unsigned int sector_count)
		: _radius(radius), _stackCount(stack_count), _sectorCount(sector_count) {

		const float PI = acos(-1);

		float x, y, z, xy;
		float nx, ny, nz, lengthInv = 1.0f / radius;
		float s, t;

		float sectorStep = 2 * PI / _sectorCount;
		float stackStep = PI / _stackCount;
		float sectorAngle, stackAngle;

		for (int i = 0; i <= _stackCount; ++i) {
			stackAngle = PI / 2 - i * stackStep;
			xy = radius * cosf(stackAngle);
			z = radius * sinf(stackAngle);

			for (int j = 0; j <= _sectorCount; ++j) {
				sectorAngle = j * sectorStep;

				x = xy * cosf(sectorAngle);
				y = xy * sinf(sectorAngle);
				_vertex.emplace_back(x, y, z);

				nx = x * lengthInv;
				ny = y * lengthInv;
				nz = z * lengthInv;

				_normal.emplace_back(nx, ny, nz);
			}
		}

		unsigned int k1, k2;
		for (int i = 0; i < _stackCount; ++i) {
			k1 = i * (_sectorCount + 1);
			k2 = k1 + _sectorCount + 1;

			for (int j = 0; j < _sectorCount; ++j, ++k1, ++k2) {
				if (i != 0) {
					_indices.emplace_back(k1, k2, k1 + 1);
				}

				if (i != (_stackCount - 1)) {
					_indices.emplace_back(k1 + 1, k2, k2 + 1);
				}
			}
		}

		_vertexCount = _vertex.size();
		_triangleCount = _indices.size();
		debug_glCheckError("Prior to Sphere Generation\n");

		glGenBuffers(1, &_VBO);
		glGenBuffers(1, &_EBO);

		glBindBuffer(GL_ARRAY_BUFFER, _VBO);
		debug_glCheckError("94");
		glBufferData(GL_ARRAY_BUFFER,
			_vertex.size() * sizeof(glm::vec3) + _normal.size() * sizeof(glm::vec3),
			nullptr,
			GL_STATIC_DRAW);
		debug_glCheckError("99");
		glBufferSubData(GL_ARRAY_BUFFER, 0, _vertex.size() * sizeof(glm::vec3), _vertex.data());
		debug_glCheckError("101");
		glBufferSubData(GL_ARRAY_BUFFER,
			_vertex.size() * sizeof(glm::vec3),
			_normal.size() * sizeof(glm::vec3),
			_normal.data());
		debug_glCheckError("106");
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof(glm::uvec3), _indices.data(), GL_STATIC_DRAW);
		debug_glCheckError("Post Sphere Generation\n");
	};

	void bind() override {
		glBindBuffer(GL_ARRAY_BUFFER, _VBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
	};

	unsigned int getVertexCount() const {
		return _vertexCount;
	};

	unsigned int getTriangleCount() const {
		return _triangleCount;
	};

private:

	float _radius;
	unsigned int _vertexCount;
	unsigned int _triangleCount;
	std::vector<glm::vec3> _vertex;
	std::vector<glm::vec3> _normal;
	std::vector<glm::uvec3> _indices;

	unsigned int _stackCount;
	unsigned int _sectorCount;

};

#endif // _AURORA_GEOMETRY_HEADER_