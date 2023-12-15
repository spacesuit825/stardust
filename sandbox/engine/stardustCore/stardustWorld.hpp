#ifndef _STARDUST_DYNAMICS_WORLD_HEADER_
#define _STARDUST_DYNAMICS_WORLD_HEADER_

#include "../stardustDynamics/stardustEngine.hpp"


namespace STARDUST {

	typedef struct WorldParameters {

	} WorldParameters;

	class World {

	public:

		
		


		void run(); // Initiate the simulation pipeline
		
	private:
		void bulkTransferDataToDevice(); // Transfers all necessary pipeline data to the GPU
		void bulkTransferDataFromDevice();// Transfers all necessary pipeline data from the GPU

		Engine* engine;

	};


	
}



#endif // _STARDUST_DYNAMICS_WORLD_HEADER_