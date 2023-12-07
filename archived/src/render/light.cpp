// Portable Light class exec

#include "light.hpp"

void Light::LogLightProperty() const {
	printf("----------------[Light Properties]-----------------");
	printf("Light POsition: [%f, %f, %f]\n", m_srcpos.x, m_srcpos.y, m_srcpos.z);
}

float* Light::getLightSrcPosFloatPtr() {
	return glm::value_ptr(m_srcpos);
}

glm::vec3 Light::getLightSrcPosVec3() const {
	return m_srcpos;
}

glm::vec3 Light::getDiffColor() const {
	return m_diffColor;
}

glm::vec3 Light::getSpecColor() const {
	return m_specColor;
}

glm::vec3 Light::getAmbColor() const {
	return m_ambColor;
}