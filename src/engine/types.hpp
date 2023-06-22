// Defining some general types

#ifndef _STARDUST_SIMULATION_TYPES_HEADER_
#define _STARDUST_SIMULATION_TYPES_HEADER_

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace STARDUST {

	using Scalar = float;

	using Vec2i = Eigen::Vector2i;
	using Vec3i = Eigen::Vector3i;
	using Vec2d = Eigen::Vector2d;
	using Vec3d = Eigen::Vector3d;
	using Vec2f = Eigen::Vector2f;
	using Vec3f = Eigen::Vector3f;

	using Mat2i = Eigen::Matrix2i;
	using Mat3i = Eigen::Matrix3i;
	using Mat2d = Eigen::Matrix2d;
	using Mat3d = Eigen::Matrix3d;
	using Mat2f = Eigen::Matrix2f;
	using Mat3f = Eigen::Matrix3f;

}

#endif // _STARDUST_SIMULATION_TYPES_HEADER_