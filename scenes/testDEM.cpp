#include <iostream>
#include <entity.hpp>
#include <renderer.hpp>
#include <engine.hpp>
#include <entity.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>


// Meshes
// Master Array - store basic data, single access per loop
// Mesh Position Array
// Mesh Quaternion Array
// Vertex Array
// Index Array

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
	float4 pos1 = make_float4(0.0f, 0.1f, 0.0f, 0.0f);
	float4 vel1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 pos2 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
	float4 pos3 = make_float4(0.0f, 0.0f, 2.0f, 0.0f);
	float4 pos4 = make_float4(0.0f, 0.0f, 3.0f, 0.0f);
	float4 pos5 = make_float4(0.0f, 0.0f, 4.0f, 0.0f);
	float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float size = 0.2;

	STARDUST::DEMParticle entity1 = STARDUST::DEMParticle(0, 1, size, size, 100, pos1, vel1); // Collider particle (set id as inactive)
	STARDUST::DEMParticle entity2 = STARDUST::DEMParticle(1, 100, size * 100, size, 9, pos2, vel2);

	//STARDUST::DEMMesh entity3 = STARDUST::DEMMesh("C:/Users/lachl/OneDrive/Documents/c++/stardust/assets/test_mesh.stl", pos3, pos5);

	engine = new STARDUST::DEMEngine(0.005);
	engine->addParticle(entity1);
	engine->addParticle(entity2);

	//engine->loadJSONSetup("C:/Users/lachl/OneDrive/Documents/c++/stardust/setup/star_test.json");

	std::cout << engine->getEntityLength();
	engine->prepArrays();
	engine->transferDataToDevice();
	engine->is_first_step = false;
	engine->transferDataToDevice();


	std::chrono::time_point<std::chrono::system_clock> start;
	std::chrono::duration<double> duration;

	double time;
	start = std::chrono::system_clock::now();

	bool check = true;

	//renderer->prepBuffers(*engine);
	//engine->bindGLBuffers(renderer->getPosVBO());

	//initGui();

	//while (!renderer->windowShouldClose()) {
	//	/*if (frame > 400 && frame % 25 == 0) {
	//		addEntity();
	//	}*/
	//	//engine->step(0.0005f);
	//	//engine->writeGLBuffers();
	//	//renderer->renderWithGUI(*engine, *gui);
	//	//handler->handleInput();

	//	check = false;
	//}

	for (int i = 0; i <= 1; i++) {
		engine->step(0.0005f);
	}

	duration = std::chrono::system_clock::now() - start;

	time = duration.count();

	std::cout << "Collision analysis completed in: " << time << "s on " << engine->getNumberOfSpheres() << " particles\n";
}
int main() {
	std::cout << "Activating Renderer... \n";
	//initRenderer();
	//initHandler();
	

	

	run();


	
	delete engine;
	delete renderer;
	delete handler;
	delete gui;

	exit(EXIT_SUCCESS);
}
