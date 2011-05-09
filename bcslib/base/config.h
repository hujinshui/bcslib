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
    #if _MSC_VER < 1500
        #error Microsoft Visual C++ of version lower than MSVC 2008 is not supported.
    #endif
    #define BCSLIB_COMPILER BCSLIB_MSVC

#elif (defined(__GNUC__))
    #if ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 3))
	    #error GCC of version lower than 4.3.0 is not supported
    #endif
	#ifdef __GXX_EXPERIMENTAL_CXX0X__
		#define BCSLIB_USE_CPP0X
	#endif
	#define BCSLIB_COMPILER BCSLIB_GCC

#else
	#error BCSLib can only be used with Microsoft Visual C++ or GCC (G++)
#endif


#ifndef BCSLIB_USE_CPP0X
	#error C++0x support is required.
#endif



/**
 *  Compiler-specific configurations
 */

#if BCSLIB_COMPILER == BCSLIB_MSVC

    #pragma warning(disable : 4996)  // suppress the warning for Microsoft's "safe" functions

    #if _MSC_VER >= 1600
        #include <stdint.h>
        #define BCS_STDINT_INCLUDED 1
    #endif
    #define BCS_PLATFORM_INTERFACE BCS_WINDOWS_INTERFACE

#elif BCSLIB_COMPILER == BCSLIB_GCC

    #include <stdint.h>
    #define BCS_STDINT_INCLUDED 1

	#define BCS_PLATFORM_INTERFACE BCS_POSIX_INTERFACE
#endif




#endif
