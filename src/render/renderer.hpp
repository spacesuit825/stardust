// Portable and Extensible Rendering Class

#ifndef _AURORA_RENDERER_HEADER_
#define _AURORA_RENDERER_HEADER_

#include <vector>
#include "graphicsentity.hpp"
#include "camera.hpp"
#include "light.hpp"
#include "shader.hpp"
#include "geometry.hpp"
#include "guiwrapper.hpp"
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "inputhandler.hpp"
//#include "../engine/solver.hpp"
//#include "../engine/rigidsolver.hpp"
#include "renderutils.hpp"

class PhysicsEntity;

class Renderer {
public:

	class Builder {
	public:

		friend class Renderer;

		Builder() = default;
		Builder& camera(glm::vec3 camera_pos, glm::vec3 lookat, glm::vec3 up = { 0., 0., 1. }, float fovy = 45, float aspect = 1,
			float z_near = 0.1, float z_far = 1000);
		Builder& light(const glm::vec3& src_pos, const glm::vec3& diff_color,
			const glm::vec3& spec_color, const glm::vec3& amb_color);
		Builder& shader(const char* vt_shader_path, const char* fg_shader_path);
		Builder& init(std::string window_name, int width = 1280, int height = 1280);

		Renderer* build();

	private:

		Renderer* m_renderer = nullptr;
		Shader* m_builder_shader = nullptr;
		Camera* m_builder_camera = nullptr;
		Light* m_builder_light = nullptr;
		GLFWwindow* m_builder_window = nullptr;
		GLuint m_builder_vao_id = 0;
		int m_builder_width;
		int m_builder_height;
	};

	~Renderer() {
		fmt::print("Destroying Renderer... \n");
		glfwDestroyWindow(m_window);
		delete m_camera;
		delete m_shader;
		delete m_light;
		
		glfwTerminate();
	}

	GLFWwindow* getWindow();

	Camera& getCamera();

	Shader& getShader();

	Light& getLight();

	inline bool windowShouldClose() { return glfwWindowShouldClose(m_window); };

	float m_background_color[4] { 158. / 256, 289. / 256, 230. / 256, 1.00f };
	float m_default_particle_color[4] { 32. / 256, 178. / 256, 170. / 256, 1.00f };
	bool m_is_use_rainbow_map = false;

	GLuint getVAO() const { return m_vao_id; };
	GLuint getPosVBO() { return m_engine_vbo_id; };
	GLuint getColorVBO() { return m_color_vbo_id; };

	//void prepBuffers(MPM::Engine& engine);
	//void renderWithGUI(MPM::Engine& engine, GUIwrapper& gui);
	void renderTest(float* particle_array, GUIwrapper& gui, int size, float dia);
	void setDefaultParticleColor(float r, float g, float b, float a = 1.0f) {
		m_default_particle_color[0] = r;
		m_default_particle_color[1] = g;
		m_default_particle_color[2] = b;
		m_default_particle_color[3] = a;
	};

	void terminate();

private:

	explicit Renderer(const Builder& builder)
		: m_window(builder.m_builder_window), m_camera(builder.m_builder_camera),
		m_light(builder.m_builder_light), m_shader(builder.m_builder_shader),
		m_vao_id(builder.m_builder_vao_id), m_sphere_mesh(0.1, 5, 10),
		m_window_width(builder.m_builder_width), m_window_height(builder.m_builder_height) {

		debug_glCheckError("Before Buffer Generation... \n");
		glGenBuffers(1, &m_engine_vbo_id);
		glGenBuffers(1, &m_color_vbo_id);
		debug_glCheckError("After Buffer Generation... \n");
	};

	Camera* m_camera = nullptr;
	Shader* m_shader = nullptr;
	Light* m_light = nullptr;
	GLFWwindow* m_window = nullptr;

	std::vector<GraphicsEntity> m_graphics_data;

	SphereMesh m_sphere_mesh;

	GLuint m_vao_id;
	GLuint m_engine_vbo_id;
	GLuint m_color_vbo_id;
	GLuint m_rigid_vbo_id;
	int m_window_width;
	int m_window_height;
	unsigned long long m_current_frame = 0;
};

#endif // _AURORA_RENDERER_HEADER_