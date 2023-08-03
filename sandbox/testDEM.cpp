#include <iostream>
#include <renderer.hpp>
#include <engine.hpp>
#include <../engine/entities/entity.hpp>
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
		.camera(glm::vec3(5., 5., 5.), glm::vec3(0, 0, 0))
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

float randFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void run() {
	float4 pos1 = make_float4(0.10f, 0.0f, 0.0f, 0.0f);
	float4 vel1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 pos2 = make_float4(0.0f, 0.0f, 2.0f, 0.0f);
	float4 pos3 = make_float4(0.50f, 0.0f, 3.0f, 0.0f);
	float4 pos4 = make_float4(0.0f, 0.0f, 3.0f, 0.0f);
	float4 pos5 = make_float4(0.0f, 0.0f, 4.0f, 0.0f);
	float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// 0, 2, 1

	float size = 0.5;

	STARDUST::DEMParticle entity1 = STARDUST::DEMParticle(0, 1, size, size, 100, pos1, vel1); // Collider particle (set id as inactive)
	//STARDUST::DEMParticle entity2 = STARDUST::DEMParticle(1, 1, size , size, 9, pos2, vel2);
	STARDUST::DEMParticle entity3 = STARDUST::DEMParticle(1, 2, size * 2, size, 9, pos3, vel2);

	//STARDUST::DEMMesh entity3 = STARDUST::DEMMesh("C:/Users/lachl/OneDrive/Documents/c++/stardust/assets/test_mesh.stl", pos3, pos5);

	STARDUST::engineConfig config = STARDUST::engineConfig{
											STARDUST::SAP,
											STARDUST::GPU
	};

	engine = new STARDUST::DEMEngine(config, 0.005);
	engine->add(entity1);
	//engine->add(entity2);
	engine->add(entity3);

	for (int i = 0; i < 1000000; i++) {

		float4 pos = make_float4(randFloat(1.0, 40.0), randFloat(1.0, 40.0), randFloat(1.0, 40.0), 0.0f);
		float4 vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		STARDUST::DEMParticle entity = STARDUST::DEMParticle(0, 1, size, size, 9, pos, vel);

		engine->add(entity);
	}

	//engine->loadJSONSetup("C:/Users/lachl/OneDrive/Documents/c++/stardust/setup/star_test.json");

	//std::cout << engine->getEntityLength()
	engine->prep();
	engine->transfer();
	//engine->transferDataToDevice();
	engine->is_first_step = false;
	//engine->transferDataToDevice();

	bool check = true;

	//renderer->prepBuffers(*engine);
	//engine->bindGLBuffers(renderer->getPosVBO());

	//initGui();

	//while (!renderer->windowShouldClose()) {

	for (int i = 0; i < 300; i++) {
		/*if (frame > 400 && frame % 25 == 0) {
			addEntity();
		}*/
		engine->update(0.0005f);
		//engine->writeGLBuffers();
		//renderer->renderWithGUI(*engine, *gui);
		//handler->handleInput();

		check = false;
	}


	std::cout << "Done... " << engine->getEntityHandler().getNumberOfSpheres() << " \n";
	//for (int i = 0; i <= 1; i++) {
	//	engine->step(0.0005f);
	//}

	
}
int main() {
	std::cout << "Activating Renderer... \n";
	//initRenderer();
	//initHandler();

	run();
	
	delete engine;
	//delete renderer;
	//delete handler;
	//delete gui;

	cudaDeviceReset();

	exit(EXIT_SUCCESS);
}
