#ifndef _STARDUST_ENGINE_HEADER_
#define _STARDUST_ENGINE_HEADER_

#include "types.hpp"
#include <entity.hpp>

namespace STARDUST {

	class DEMEngine {
		// Class to setup and initate simulations
	public:

		DEMEngine(Scalar test) : m_domain(test) {};

		//void loadXMLSetup(std::string); // Loads setup XML file and creates the scene
		void loadJSONSetup(std::string); // Loads setup JSON file and creates the scene

		//void createEntity(int, std::string, int); // Load mesh from file and create n entities // n, filepath, grid_resolution
		//void createEntity(int, Scalar, int); // Create n generic entities from basic inputs // n, size of cube, grid_resolution

		//void cloneEntity(int); // Clone a specific entity n times, spacing is randomised by the BB of the entity

		//void prepArrays(); // Prepare std::vectors for CUDA by translating into naked arrays
		//void initCUDA(); // Mallocate space on the GPU for the maximum number of particles/entities
		//void copyHostToDevice(); // Initiate memCopy of Host data to the GPU

		//void step(Scalar); // Perform single step in time

		int getEntityLength() { return m_entities.size(); };

	private:

		Scalar m_domain;


		std::vector<DEMEntity> m_entities;
	};
}



#endif // _STARDUST_ENGINE_HEADER_