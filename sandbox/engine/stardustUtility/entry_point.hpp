#ifndef _STARDUST_ENTRY_POINT_HEADER_
#define _STARDUST_ENTRY_POINT_HEADER_

#include "application.hpp"

int main(int argc, char** argv) {

	// Create an instance of the STARDUST application
	auto app = STARDUST::InitiateApplication(argc, argv);

	// Start the instance
	app->run();

	// Destroy the instance
	delete app;

}

#endif // _STARDUST_ENTRY_POINT_HEADER_