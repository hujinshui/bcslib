/**
 * @file cmath.h
 *
 * Proper inclusion of math functions.
 * 
 * Purposes:
 * 	1. MSVC lacks a great number of std C math functions, need to workarond it.
 * 	2. Make it easier to switch to more efficient math libs in future.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#include <bcslib/base/basic_defs.h>
#include <cmath>
#include <type_traits>

#ifndef BCSLIB_CMATH_H_
#define BCSLIB_CMATH_H_

namespace bcs
{
	namespace math
	{
		// inclusion C++03 standard ones

		using std::abs;
		using std::sqrt;
		using std::pow;

		using std::exp;
		using std::log;
		using std::log10;
		using std::frexp;
		using std::ldexp;

		using std::floor;
		using std::ceil;
		using std::fmod;

		using std::sin;
		using std::cos;
		using std::tan;
		using std::asin;
		using std::acos;
		using std::atan;
		using std::atan2;

		using std::sinh;
		using std::cosh;
		using std::tanh;

		// BCSLib's extras (for convenience)

		template<typename T>
		BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
		sqr(T x)
		{
			return x * x;
		}

		template<typename T>
		BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
		cube(T x)
		{
			return x * x * x;
		}

		template<typename T>
		BCS_FORCE_INLINE typename std::enable_if<std::is_floating_point<T>::value, T>::type
		rcp(T x)
		{
			return T(1) / x;
		}

		template<typename T>
		BCS_FORCE_INLINE typename std::enable_if<std::is_floating_point<T>::value, T>::type
		rsqrt(T x)
		{
			return T(1) / (x * x);
		}

		template<typename T>
		BCS_FORCE_INLINE T rgn_bound(T x, T lb, T ub)
		{
			return x < lb ? lb : (x > ub ? ub : x);
		}

		// Part of the new stuff in C++0x (Unfortunately, many are not available for MSVC yet)

#ifdef BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE

		using std::cbrt;
		using std::copysign;
		using std::hypot;

		using std::exp2;
		using std::log2;
		using std::logb;
		using std::expm1;
		using std::log1p;

		using std::round;
		using std::trunc;

		using ::sincos;

		using std::asinh;
		using std::acosh;
		using std::atanh;

		using std::erf;
		using std::erfc;
		using std::lgamma;
		using std::tgamma;

		using std::isfinite;
		using std::isinf;
		using std::isnan;
		using std::signbit;

#else
		// TODO: implement C99 math for MSVC
#error Sorry, yet to implement the C99 math for MSVC.
#endif


	}
}

#endif
