#include <iostream>
#include <entity.hpp>
#include <renderer.hpp>
#include <engine.hpp>

#include <renderer.hpp>

Renderer* renderer = nullptr;
InputHandler* handler = nullptr;
GUIwrapper* gui = nullptr;

STARDUST::DEMEntity entity = STARDUST::DEMEntity(0, 5, 0.05, STARDUST::Vec3f(1, 1, 1), STARDUST::Vec3f(0, 0, 0));

STARDUST::Scalar* particle_array = entity.convertParticlePositionsToNakedArray();

std::vector<STARDUST::DEMParticle> particles = entity.getParticles();

int size = particles.size();

STARDUST::Scalar diameter = particles[0].size;



void initRenderer() {
	renderer = Renderer::Builder()
		.init("Test Scene", 1000, 800)
		.camera(glm::vec3(3., 3., 3.), glm::vec3(0, 0, 0))
		.shader("C:/Users/lachl/OneDrive/Documents/c++/aurora/src/render/shader/vertex.glsl", "C:/Users/lachl/OneDrive/Documents/c++/aurora/src/render/shader/fragment.glsl")
		.light(glm::vec3(0.5, 0.5, 0.5),
			glm::vec3(1., 1., 1.),
			glm::vec3(0.1, 0.1, 0.1),
			glm::vec3(0, 0, 0))
		.build();
}

void initHandler() {
	handler = new InputHandler(renderer);
}

void initGui() {
	gui = new GUIwrapper();

	(*gui)
		.init(renderer->getWindow())
		.startGroup("App")
		.addWidgetText("App FPS: %.1f", gui->m_frame_rate)
		.endGroup()
		.build();
}

void run() {
	while (!renderer->windowShouldClose()) {
		renderer->renderTest(particle_array, *gui, size, diameter);
		handler->handleInput();
	}
}
int main() {
	std::cout << "Activating Renderer... \n";
	initRenderer();
	initHandler();
	initGui();

	STARDUST::DEMEngine engine = STARDUST::DEMEngine(0.4f);
	engine.loadJSONSetup("C:/Users/lachl/OneDrive/Documents/c++/stardust/setup/star_test.json");

	std::cout << engine.getEntityLength();

	run();

	std::cout << "Terminating Renderer... \n";
	delete renderer;
	delete handler;
	delete gui;

	exit(EXIT_SUCCESS);
}
