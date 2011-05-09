/**
 * @file config.h
 *
 * The configuration file for Basic Computation Supporting Library
 *
 * @author dhlin
 */

#ifndef BCSLIB_CONFIG_H
#define BCSLIB_CONFIG_H


#define BCSLIB_MSVC 0x01
#define BCSLIB_GCC 0x02

#define BCS_WINDOWS_INTERFACE 0x11
#define BCS_POSIX_INTERFACE 0x12


#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
	#if _MSC_VER < 1600
	#error Microsoft Visual C++ of version lower than MSVC 2010 is not supported.
	#endif
	#define BCSLIB_COMPILER BCSLIB_MSVC

	#define BCS_PLATFORM_INTERFACE BCS_WINDOWS_INTERFACE

#elif (defined(__GNUC__))
	#if ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 5))
		#error GCC of version lower than 4.5.0 is not supported
	#endif
	#define BCSLIB_COMPILER BCSLIB_GCC

	#define BCS_PLATFORM_INTERFACE BCS_POSIX_INTERFACE

#else
	#error BCSLib can only be used with Microsoft Visual C++ or GCC (G++)
#endif


#define BCS_TR1_INCLUDE_STD_DIR 0
#define BCS_TR1_INCLUDE_TR1_DIR 1
#define BCS_TR1_INCLUDE_BOOST_DIR 2


#if (BCSLIB_COMPILER == BCSLIB_MSVC)

	#define BCSLIB_TR1_INCLUDE_DIR BCS_TR1_INCLUDE_STD_DIR
	#define BCS_TR1_FROM_NAMESPACE std::tr1

#elif (BCSLIB_COMPILER == BCSLIB_GCC)

	#ifdef __GXX_EXPERIMENTAL_CXX0X__
		#define BCSLIB_TR1_INCLUDE_DIR BCS_TR1_INCLUDE_STD_DIR
		#define BCS_TR1_FROM_NAMESPACE std
	#else
		#define BCSLIB_TR1_INCLUDE_DIR BCS_TR1_INCLUDE_TR1_DIR
		#define BCS_TR1_FROM_NAMESPACE std::tr1
	#endif

#endif


#endif

