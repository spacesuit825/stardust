// Renderer exec

#include "renderer.hpp"
#include <gl/glew.h>
#include <GLFW/glfw3.h>

#include "graphicsentity.hpp"
#include "guiwrapper.hpp"
#include <iostream>

Camera& Renderer::getCamera() {
	return *m_camera;
}

//void Renderer::prepBuffers(MPM::Engine& engine) {
//	glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
//	glBufferData(GL_ARRAY_BUFFER,
//		3 * engine.getEngineConfig().max_particles * sizeof(float),
//		0,
//		GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//}

//void Renderer::renderWithGUI(MPM::Engine& engine, GUIwrapper& gui) {
//	glBindVertexArray(m_vao_id);
//
//	debug_glCheckError("Render Loop Initialized");
//
//	// Setup renderer
//	glClearColor(m_background_color[0], m_background_color[1], m_background_color[2], 1.0f);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	// Set the camera properties
//	m_shader->setUniform("eyepos", m_camera->getCameraPos());
//
//	// Set the Lighting properties
//	m_shader->setUniform("lightsrc", m_light->getLightSrcPosVec3());
//	m_shader->setUniform("Sd", m_light->getDiffColor());
//	m_shader->setUniform("Ss", m_light->getSpecColor());
//	m_shader->setUniform("Sa", m_light->getAmbColor());
//	debug_glCheckError("Light Property Error\n");
//
//	// Set Particle properties
//	m_shader->setUniform("isUseRainBowMap", m_is_use_rainbow_map);
//	m_shader->setUniform("Kd",
//		glm::vec3(m_default_particle_color[0],
//			m_default_particle_color[1],
//			m_default_particle_color[2]));
//	m_shader->setUniform("Ka", glm::vec3(0.0, 0.0, 0.0));
//	m_shader->setUniform("Ks", glm::vec3(0.1, 0.1, 0.1));
//	m_shader->setUniform("Ke", glm::vec3(0, 0, 0));
//	m_shader->setUniform("sh", 0.1f);
//	m_shader->setUniform("particle_scale", m_particle_scale);
//
//	// Set model/view/proj matrices
//	m_shader->setUniform("modelMat", glm::mat4(1.0f));
//	m_shader->setUniform("viewMat", m_camera->getViewMatrix());
//	m_shader->setUniform("projMat", m_camera->getProjectionMatrix());
//
//	// Binding particle sphere vbo
//	m_sphere_mesh.bind();
//	glEnableVertexAttribArray(0);
//	// Pointer for vertex positions
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//	glEnableVertexAttribArray(1);
//	// Pointer for color for each vertex/triangle
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
//		(void*)(m_sphere_mesh.getVertexCount() * sizeof(glm::vec3)));
//
//	// Bind the particle positions vbo and reserve the required space 
//	// ie. 3 * particle_number (float) == particle positions (x, y, z)
//	// and 1 * particle_number (float) == color weight (single float)
//
//	// TODO: move this outside the render loop
//	//glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
//	//glBufferData(GL_ARRAY_BUFFER,
//	//	3 * engine.getParticleCount() * sizeof(float),
//	//	0,
//	//	GL_DYNAMIC_DRAW);
//
//	//// Get rid of the color weights, dumb idea to begin with
//
//	//// Occupy reserved space with particles positions
//	//glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * engine.getParticleCount() * sizeof(float),
//	//	engine.getParticlePosPtr());
//	// Sub in remaining space with color weighting floats
//	/*glBufferSubData(GL_ARRAY_BUFFER,
//		3 * engine.getParticleCount() * sizeof(float),
//		engine.getParticleCount() * sizeof(float),
//		engine.mCurrentParticleColorWeight.data());*/
//
//	glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
//
//	// Vertex Pointer to positions (ie. 0 offset and stepping 3 * float for each position)
//	glEnableVertexAttribArray(2);
//	glVertexAttribDivisor(2, 1);
//	glVertexAttribPointer(2, 
//		3, 
//		GL_FLOAT, 
//		GL_FALSE, 
//		3 * sizeof(float), 
//		(void*)0
//	);
//
//	//// Vertex Pointer to color weights (ie. 3 * particle_number * float offset and stepping of 1 float)
//	//glEnableVertexAttribArray(3);
//	//glVertexAttribDivisor(3, 1);
//	//glVertexAttribPointer(3,
//	//	1,
//	//	GL_FLOAT,
//	//	GL_FALSE,
//	//	sizeof(float),
//	//	(void*)(3 * engine.getParticleCount() * sizeof(float)));
//
//	// For colors
//	glBindBuffer(GL_ARRAY_BUFFER, m_color_vbo_id);
//	glBufferData(GL_ARRAY_BUFFER,
//		3 * engine.getParticleCount() * sizeof(float),
//		nullptr,
//		GL_DYNAMIC_DRAW);
//
//	glBufferSubData(GL_ARRAY_BUFFER, 
//		0, 
//		3 * engine.getParticleCount() * sizeof(float), 
//		engine.getColorPtr());
//
//	glEnableVertexAttribArray(3);
//	glVertexAttribDivisor(3, 1);
//	glVertexAttribPointer(3,
//		3,
//		GL_FLOAT,
//		GL_FALSE,
//		3 * sizeof(float),
//		(void*)0
//	);
//
//	glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
//
//	// Renders spheres at each location based on instanced arrays. This saves on GPU draw calls,
//	// and drastically accelerates rendering.
//	
//	// The divisor calls on the position and color weight arrays, indicates to OpenGL that 
//	// for each new instance of a sphere a new position and color weight should be retrieved and
//	// passed to the vertex shader. If the divisor had 0 set, a new position would be retrieved
//	// for every vertex of the sphere and we would have a mangled mess.
//	glDrawElementsInstanced(GL_TRIANGLES, m_sphere_mesh.getTriangleCount() * 3,
//		GL_UNSIGNED_INT, nullptr, engine.getParticleCount());
//
//	
//	glfwPollEvents();
//	gui.render();
//	int display_w, display_h;
//	glfwGetFramebufferSize(m_window, &display_w, &display_h);
//	glViewport(0, 0, display_w, display_h);
//	glfwSwapBuffers(m_window);
//
//	glBindVertexArray(0);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//}

Light& Renderer::getLight() {
	return *m_light;
}

Shader& Renderer::getShader() {
	return *m_shader;
}

GLFWwindow* Renderer::getWindow() {
	return m_window;
}

void Renderer::terminate() {
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	delete (this);
}

Renderer::Builder& Renderer::Builder::camera(glm::vec3 camera_pos,
	glm::vec3 lookat,
	glm::vec3 up,
	float fovy,
	float aspect,
	float z_near,
	float z_far) {

	m_builder_camera = new Camera(camera_pos, lookat, up, fovy, aspect, z_near, z_far);
	return *this;
}

Renderer::Builder &
Renderer::Builder::light(const glm::vec3& src_pos, const glm::vec3& diff_color,
	const glm::vec3& spec_color, const glm::vec3& amb_color) {

	m_builder_light = new Light(src_pos, diff_color, spec_color, amb_color);
	return *this;
}

Renderer* Renderer::Builder::build() {
	m_renderer = new Renderer(*this);

	return m_renderer;
}

Renderer::Builder& Renderer::Builder::init(std::string window_name, int width, int height) {

#define GLEW_STATIC

	m_builder_width = width;
	m_builder_height = height;
	int m_is_glfw_init = glfwInit();
	if (!m_is_glfw_init) {
		std::cout << "GLFW Init Failed\n";
	}

#if defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	m_builder_window = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
	if (m_builder_window == nullptr) {
		std::cout << "Window Generation Failed\n";
	}

	glfwMakeContextCurrent(m_builder_window);
	glfwSwapInterval(0);

	bool err = glewInit() != GLEW_OK;

	if (err) {
		fprintf(stderr, "Failed to initialize OpenGL loader\n");
	}

	glGenVertexArrays(1, &m_builder_vao_id);
	glBindVertexArray(m_builder_vao_id);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	debug_glCheckError("Post Renderer Init");
	return *this;
};

Renderer::Builder& Renderer::Builder::shader(const char* vt_shader_path, const char* fg_shader_path) {
	m_builder_shader = new Shader(vt_shader_path, fg_shader_path);
	m_builder_shader->use();
	return *this;
}

void Renderer::renderTest(float* particle_array, GUIwrapper& gui, int size, float dia) {
	glBindVertexArray(m_vao_id);

	debug_glCheckError("Render Loop Initialized");

	// Setup renderer
	glClearColor(m_background_color[0], m_background_color[1], m_background_color[2], 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set the camera properties
	m_shader->setUniform("eyepos", m_camera->getCameraPos());

	// Set the Lighting properties
	m_shader->setUniform("lightsrc", m_light->getLightSrcPosVec3());
	m_shader->setUniform("Sd", m_light->getDiffColor());
	m_shader->setUniform("Ss", m_light->getSpecColor());
	m_shader->setUniform("Sa", m_light->getAmbColor());
	debug_glCheckError("Light Property Error\n");

	// Set Particle properties
	m_shader->setUniform("isUseRainBowMap", m_is_use_rainbow_map);
	m_shader->setUniform("Kd",
		glm::vec3(m_default_particle_color[0],
			m_default_particle_color[1],
			m_default_particle_color[2]));
	m_shader->setUniform("Ka", glm::vec3(0.0, 0.0, 0.0));
	m_shader->setUniform("Ks", glm::vec3(0.1, 0.1, 0.1));
	m_shader->setUniform("Ke", glm::vec3(0, 0, 0));
	m_shader->setUniform("sh", 0.01f);
	m_shader->setUniform("particle_scale", dia*10);

	// Set model/view/proj matrices
	m_shader->setUniform("modelMat", glm::mat4(1.0f));
	m_shader->setUniform("viewMat", m_camera->getViewMatrix());
	m_shader->setUniform("projMat", m_camera->getProjectionMatrix());

	// Binding particle sphere vbo
	m_sphere_mesh.bind();
	glEnableVertexAttribArray(0);
	// Pointer for vertex positions
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	// Pointer for color for each vertex/triangle
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
		(void*)(m_sphere_mesh.getVertexCount() * sizeof(glm::vec3)));

	// Bind the particle positions vbo and reserve the required space 
	// ie. 3 * particle_number (float) == particle positions (x, y, z)
	// and 1 * particle_number (float) == color weight (single float)

	// TODO: move this outside the render loop
	glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);
	glBufferData(GL_ARRAY_BUFFER,
		size * 3 * sizeof(float),
		nullptr,
		GL_DYNAMIC_DRAW);

	debug_glCheckError("Bind Buffers...");

	//std::cout << "array size " << size;
	// Occupy reserved space with particles positions
	glBufferSubData(GL_ARRAY_BUFFER, 0, size * 3 * sizeof(float), particle_array);

	debug_glCheckError("Sub Data...");

	// Vertex Pointer to positions (ie. 0 offset and stepping 3 * float for each position)
	glEnableVertexAttribArray(2);
	glVertexAttribDivisor(2, 1);
	glVertexAttribPointer(2, 
		3, 
		GL_FLOAT, 
		GL_FALSE, 
		3 * sizeof(float), 
		(void*)0
	);

	debug_glCheckError("Vertex pointers...");

	/*float color_array[12] = { 0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f, 0.4f, 0.3f, 0.2f };

	glBindBuffer(GL_ARRAY_BUFFER, m_color_vbo_id);
	glBufferData(GL_ARRAY_BUFFER,
		3 * 4 * sizeof(float),
		nullptr,
		GL_DYNAMIC_DRAW);

	glBufferSubData(GL_ARRAY_BUFFER,
		0,
		3 * 4 * sizeof(float),
		color_array);

	glEnableVertexAttribArray(3);
	glVertexAttribDivisor(3, 1);
	glVertexAttribPointer(3,
		3,
		GL_FLOAT,
		GL_FALSE,
		3 * sizeof(float),
		(void*)0
	);*/

	glBindBuffer(GL_ARRAY_BUFFER, m_engine_vbo_id);

	// Renders spheres at each location based on instanced arrays. This saves on GPU draw calls,
	// and drastically accelerates rendering.

	// The divisor calls on the position and color weight arrays, indicates to OpenGL that 
	// for each new instance of a sphere a new position and color weight should be retrieved and
	// passed to the vertex shader. If the divisor had 0 set, a new position would be retrieved
	// for every vertex of the sphere and we would have a mangled mess.
	glDrawElementsInstanced(GL_TRIANGLES, m_sphere_mesh.getTriangleCount() * 3,
		GL_UNSIGNED_INT, nullptr, size);


	glfwPollEvents();
	gui.render();
	int display_w, display_h;
	glfwGetFramebufferSize(m_window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glfwSwapBuffers(m_window);

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
