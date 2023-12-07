// Portable Camera class exec

// Note: I have chosen the z-axis as the upwards direction


#include "camera.hpp"

// Moving Camera function definitions

void Camera::moveUp() {
	m_camera_pos = m_camera_pos + m_t_sensitivity * glm::vec3(0, 0, 1);
	updateViewMatrix();
}

void Camera::moveDown() {
	m_camera_pos = m_camera_pos - m_t_sensitivity * glm::vec3(0, 0, 1);
	updateViewMatrix();
}

void Camera::moveFront() {
	m_camera_pos = m_camera_pos + m_t_sensitivity * m_camera_front;
	updateViewMatrix();
}

void Camera::moveBack() {
	m_camera_pos = m_camera_pos - m_t_sensitivity * m_camera_front;
	updateViewMatrix();
}

void Camera::moveRight() {
	m_camera_pos = m_camera_pos + m_t_sensitivity * m_camera_right;
	updateViewMatrix();
}

void Camera::moveLeft() {
	m_camera_pos = m_camera_pos - m_t_sensitivity * m_camera_right;
	updateViewMatrix();
}

void Camera::rotateYaw(float degree) {
	float theta = glm::radians(m_r_sensitivity * degree);
	glm::quat q = glm::angleAxis(theta, m_camera_up);

	m_camera_right = q * m_camera_right;
	m_camera_right.z = 0;

	m_camera_up = glm::normalize(m_camera_up);
	m_camera_front = glm::cross(m_camera_up, m_camera_right);

	updateViewMatrix();
}

void Camera::rotatePitch(float degree) {
	float theta = glm::radians(m_r_sensitivity * degree);
	glm::quat q = glm::angleAxis(theta, m_camera_right);

	m_camera_front = q * m_camera_front;
	m_camera_up = glm::cross(m_camera_right, m_camera_front);

	updateViewMatrix();
}

//void logCameraProperty() const {
	//fmt::print("---------------[Camera Properties]---------------\n");
	//fmt::format("Camera Position: [{}, {}, {}]\n", m_camera_pos.x, m_camera_pos.y, m_camera_pos.z);
	//fmt::print("Field of View: {}\n", m_fovy);
	//fmt::print("Aspect Ratio: {}\n", m_aspect);
//}

void Camera::updateViewMatrix() {
	m_view_matrix = glm::lookAt(
		m_camera_pos,
		m_camera_pos + m_camera_front,
		m_camera_up
	);
}

void Camera::updateProjectionMatrix() {
	m_projection_matrix = glm::perspective(
		glm::radians(m_fovy),
		m_aspect,
		m_z_near,
		m_z_far
	);
}