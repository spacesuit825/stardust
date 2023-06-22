// Portable light class header

#ifndef _AURORA_LIGHT_HEADER_
#define _AURORA_LIGHT_HEADER_

#include <gl/glew.h>
#include <glm.hpp>
#include <gtc/type_ptr.hpp>
#include <cstdio>

class Light {
public:

	Light(const glm::vec3& src_pos, const glm::vec3& diff_color, const glm::vec3& spec_color, const glm::vec3& amb_color) :
		m_srcpos(src_pos), m_diffColor(diff_color), m_specColor(spec_color), m_ambColor(amb_color) {

	};

	float* getLightSrcPosFloatPtr();
	glm::vec3 getLightSrcPosVec3() const;

	glm::vec3 getDiffColor() const;
	glm::vec3 getSpecColor() const;
	glm::vec3 getAmbColor() const;
	
	void LogLightProperty() const;

private:

	glm::vec3 m_srcpos;
	glm::vec3 m_diffColor;
	glm::vec3 m_specColor;
	glm::vec3 m_ambColor;
};

#endif // _AURORA_LIGHT_HEADER_
