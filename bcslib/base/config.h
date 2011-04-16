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


#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
        #if _MSC_VER < 1500
                #error Microsoft Visual C++ of version lower than MSVC 2008 is not supported.
        #endif
        #define BCSLIB_COMPILER BCSLIB_MSVC
#elif (defined(__GNUC__))
        #if __GNUC__ < 4
                #error GCC (G++) of version lower than 4.0.0 is not supported
        #endif
        #define BCSLIB_COMPILER BCSLIB_GCC
#else
        #error BCSLib can only be used with Microsoft Visual C++ or GCC (G++)
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

#elif BCSLIB_COMPILER == BCSLIB_GCC

        #include <stdint.h>
        #define BCS_STDINT_INCLUDED 1

#endif




#endif
