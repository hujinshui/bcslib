/**
 * @file config.h
 *
 * The configuration file for Basic Computation Supporting Library
 *
 * @author dhlin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_CONFIG_H
#define BCSLIB_CONFIG_H

/************************************************
 *
 *  User controls
 *
 ************************************************/

/**
 * Whether to use SSE2:
 *
 * Note: SSE2 contains a large set of useful vectorized
 * computation instructions
 */
#define BCSLIB_USE_SSE2

/**
 * Whether to use SSE3:
 *
 * Note: SSE3 contains HADDPS and HADDPD (useful to speed up sum)
 */
#define BCSLIB_USE_SSE3


/**
 * Whether to use SSE4.1:
 *
 * Note: SSE4.1 contains DPPS and DPPD (useful to speed up dot product)
 */
#define BCSLIB_USE_SSE41


/**
 * Whether to turn off extensive checks (e.g. array bound)
 */
// #define BCSLIB_NO_DEBUG



/************************************************
 *
 *  Compiler detection
 *
 ************************************************/

#define BCSLIB_MSVC 0x01
#define BCSLIB_GCC 0x02
#define BCSLIB_CLANG 0x03

#define BCS_WINDOWS_INTERFACE 0x11
#define BCS_POSIX_INTERFACE 0x12

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
	#if _MSC_VER < 1600
		#error Microsoft Visual C++ of version lower than MSVC 2010 is not supported.
	#endif
	#define BCSLIB_COMPILER BCSLIB_MSVC

	#define BCS_PLATFORM_INTERFACE BCS_WINDOWS_INTERFACE

	#define BCS_USE_C11_STDLIB
	#define BCS_USE_STATIC_ASSERT

#elif (defined(__GNUC__))

	#define BCS_HAS_C99_MATH

	#if (defined(__clang__))
		#if ((__clang_major__ < 2) || (__clang_major__ == 2 && __clang_minor__ < 8))
			#error CLANG of version lower than 2.8.0 is not supported
		#endif
		#define BCSLIB_COMPILER BCSLIB_CLANG

		#define BCS_USE_C11_STDLIB
		#define BCS_USE_STATIC_ASSERT

	#else
		#if ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 2))
			#error GCC of version lower than 4.2.0 is not supported
		#endif
		#define BCSLIB_COMPILER BCSLIB_GCC

		#if (defined(__GXX_EXPERIMENTAL_CXX0X__))
			#define BCS_USE_C11_STDLIB
			#define BCS_USE_STATIC_ASSERT
		#endif
	#endif

	#define BCS_PLATFORM_INTERFACE BCS_POSIX_INTERFACE

#else
	#error BCSLib can only be used with Microsoft Visual C++, GCC (G++), or clang (clang++).
#endif


#ifdef BCS_USE_C11_STDLIB
	#define BCS_TR1 std
#else
	#define BCS_TR1 std::tr1
#endif


#endif

