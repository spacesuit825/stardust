// Internal
#include "engine.hpp"

// C++
#include <string>
#include <iostream>
#include <fstream>

// External
#include <json.hpp>

using json = nlohmann::json;

namespace STARDUST {

	void DEMEngine::loadJSONSetup(std::string filepath) {
		
		std::ifstream jsonfile(filepath);
		std::stringstream buffer;
		buffer << jsonfile.rdbuf();
		auto data = json::parse(buffer);

		// Set domain size in metres
		// TODO: Add option for unbounded
		m_domain = (Scalar)data["scene"]["domain"];

		// Load a series of entities from JSON
		json entities = data["scene"]["entities"];
		for (auto entity_data : entities) {
			if (entity_data["type"] == "generic") {
				Vec3f pos = Vec3f(
					(Scalar)entity_data["position"][0],
					(Scalar)entity_data["position"][1],
					(Scalar)entity_data["position"][2]
				);

				Vec3f vel = Vec3f(
					(Scalar)entity_data["velocity"][0],
					(Scalar)entity_data["velocity"][1],
					(Scalar)entity_data["velocity"][2]
				);

				Scalar size = (Scalar)entity_data["size"];

				int length = m_entities.size();

				DEMEntity entity = DEMEntity(length + 1, 5, size, pos, vel);
				m_entities.push_back(entity);
			} else if (entity_data["type"] == "mesh") {

			}
		}



	}
}