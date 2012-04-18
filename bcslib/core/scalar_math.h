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

#include <bcslib/core/basic_types.h>
#include <bcslib/core/syntax.h>
#include <cmath>

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
		inline BCS_ENSURE_INLINE T sqr(T x)
		{
			return x * x;
		}

		template<typename T>
		inline BCS_ENSURE_INLINE T cube(T x)
		{
			return x * x * x;
		}

		template<typename T>
		inline BCS_ENSURE_INLINE T rcp(T x)
		{
			return T(1) / x;
		}

		template<typename T>
		inline BCS_ENSURE_INLINE T rsqrt(T x)
		{
			return T(1) / sqrt(x);
		}
	}

	// Comparison

	template<typename T>
	inline BCS_ENSURE_INLINE T clamp(T x, T lb, T ub)
	{
		return x < lb ? lb : (x > ub ? ub : x);
	}

	template<typename T>
	inline BCS_ENSURE_INLINE const T& min(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	inline BCS_ENSURE_INLINE const T& max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	inline BCS_ENSURE_INLINE T& lbound(T &x, const T &lb)
	{
		if (x < lb) x = lb;
		return x;
	}

	template<typename T>
	inline BCS_ENSURE_INLINE T& ubound(T &x, const T &ub)
	{
		if (x > ub) x = ub;
		return x;
	}

}

#endif
