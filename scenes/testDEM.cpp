#include <iostream>
#include <entity.hpp>
#include <renderer.hpp>
#include <engine.hpp>

#include <renderer.hpp>

STARDUST::DEMEngine* engine = nullptr;
Renderer* renderer = nullptr;
InputHandler* handler = nullptr;
GUIwrapper* gui = nullptr;


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
		//renderer->renderTest(particle_array, *gui, size, diameter);
		//handler->handleInput();
	}
}
int main() {
	std::cout << "Activating Renderer... \n";
	initRenderer();
	initHandler();
	initGui();

	engine = new STARDUST::DEMEngine(0.4f);
	engine->loadJSONSetup("C:/Users/lachl/OneDrive/Documents/c++/stardust/setup/star_test.json");

	std::cout << engine->getEntityLength();
	engine->prepArrays();
	engine->transferDataToDevice();
	engine->is_first_step = false;
	engine->transferDataToDevice();

	//run();
	
	delete engine;
	delete renderer;
	delete handler;
	delete gui;

	exit(EXIT_SUCCESS);
}
