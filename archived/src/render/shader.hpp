// Portable Shader class header

#ifndef _AURORA_SHADER_HEADER_
#define _AURORA_SHADER_HEADER_

#include <string>
#include <gl/glew.h>
#include <glm.hpp>

class Shader {
public:

	// Shader constructor
	Shader(const char* vt_shader_path, const char* fg_shader_path)
		: m_is_source_loaded(false), m_is_source_compiled(false), m_is_program_made(false) {

		m_vertex_shader_path = vt_shader_path;
		m_fragment_shader_path = fg_shader_path;

		m_vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
		m_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

		this->m_is_source_loaded = loadSource();
		this->m_is_source_compiled = compile();
		this->m_is_program_made = makeProgram();
	};

	// Shader destructor
	~Shader() { glDeleteProgram(m_program_id); };

	void use();

	// Retreival Methods
	GLuint getProgramID();
	GLuint getUniformLocation(const char* t_name) const;

	// Set Methods for GLSL shaders
	void setUniform(const char* t_name, glm::vec3 t_v3);
	void setUniform(const char* t_name, glm::mat4 t_m4);
	void setUniform(const char* t_name, glm::mat3 t_m3);
	void setUniform(const char* t_name, float t_f);
	void setUniform(const char* t_name, bool t_b);

private:
	// Flexibilty to allow either file saved GLSL shaders or raw GLSL code

	// GLSL file path to load 
	std::string m_vertex_shader_path;
	std::string m_fragment_shader_path;

	// Raw GLSL code in string format
	std::string m_vertex_shader_code;
	std::string m_fragment_shader_code;

	// Intrinsic shader properties
	GLuint m_program_id;
	GLuint m_vertex_shader_id;
	GLuint m_fragment_shader_id;
	
	bool m_is_source_loaded;
	bool m_is_source_compiled;
	bool m_is_program_made;

	// Shader initialisation
	int loadSource();
	int compile();
	int makeProgram();
	
	// Setting uniforms within GLSL program (change color, etc.)
	int setUniform();
};

#endif // _AURORA_SHADER_HEADER_