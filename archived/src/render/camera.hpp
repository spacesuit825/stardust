// Portable Camera Class Header

#ifndef _AURORA_CAMERA_HEADER_
#define _AURORA_CAMERA_HEADER_

// GLM math helper
#include <glm.hpp>
#include <gtx/transform.hpp>
#include <gtc/quaternion.hpp>
#include <gtc/type_ptr.hpp>


// FMT prompt helper
#include <fmt/core.h>

class Camera {
public:

	// Camera contructor
	Camera(glm::vec3 camerapos, glm::vec3 lookat, glm::vec3 up, float fovy,
		float aspect, float z_near, float z_far) :
		m_camera_pos(camerapos), m_fovy(fovy), m_aspect(aspect),
		m_z_near(z_near), m_z_far(z_far) {

		// Camera space setup
		m_camera_front = glm::normalize(lookat - m_camera_pos);
		m_camera_right = glm::normalize(glm::cross(m_camera_front, up));
		m_camera_up = glm::normalize(glm::cross(m_camera_right, m_camera_front));

		// View Matrix
		m_view_matrix = glm::lookAt(
			m_camera_pos,
			lookat,
			up
		);

		// Projection matrix
		m_projection_matrix = glm::perspective(
			glm::radians(m_fovy),
			m_aspect,
			m_z_near,
			m_z_far
		);

	};

	// Retrieve intrinsic matrices/properties
	inline glm::mat4 getViewMatrix() { return m_view_matrix; };
	inline glm::mat4 getProjectionMatrix() { return m_projection_matrix; };
	inline glm::vec3 getCameraPos() { return m_camera_pos; };
	inline float* getCameraPosFloatPtr() { return glm::value_ptr(m_camera_pos); };
	inline float getFovy() { return m_fovy; };

	// Set views/sensitivities
	inline void setCameraTranslationalSensitivity(float s) { this->m_t_sensitivity = s; };
	inline void setCameraRotationalSensitivity(float s) { this->m_r_sensitivity = s; };
	inline void setFovy(float fovy) {
		this->m_fovy = fovy;
		updateProjectionMatrix();
	};

	// Moving the camera in space
	void moveUp();

	void moveDown();

	void moveFront();

	void moveBack();
	
	void moveRight();

	void moveLeft();

	void rotateYaw(float degree);

	void rotatePitch(float degree);

	// for tracking camera in space, if necessary
	//void logCameraProperty() const; //

	float m_t_sensitivity = 0.05;
	float m_r_sensitivity = 0.025;

private:

	// Intrinsic Camera properties
	glm::vec3 m_camera_pos;
	glm::vec3 m_camera_up;
	glm::vec3 m_camera_front;
	glm::vec3 m_camera_right;

	float m_z_near;
	float m_z_far;
	float m_aspect;
	float m_fovy;

	glm::mat4 m_projection_matrix;
	glm::mat4 m_view_matrix;

	void updateViewMatrix();
	void updateProjectionMatrix();
};

#endif // _AURORA_CAMERA_HEADER_


