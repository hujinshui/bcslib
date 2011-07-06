/**
 * @file arbb_linalg.h
 *
 * The linear algebraic computation based on ArBB
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARBB_LINALG_H_
#define BCSLIB_ARBB_LINALG_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>

#include <arbb.hpp>

namespace bcs
{
	/********************************************
	 *
	 *  vector norms
	 *
	 ********************************************/

	// L1 norm

	template<typename T>
	inline T norm_L1(const arbb::dense<T, 1>& a)
	{
		return arbb::add_reduce(arbb::abs(a));
	}

	template<typename T, size_t D>
	inline T norm_L1(const arbb::dense<T, D>& a)
	{
		return norm_L1(a.flatten());
	}

	template<typename T>
	inline arbb::dense<T, 1> norm_L1(const arbb::dense<T, 2>& a, unsigned int level)
	{
		return arbb::add_reduce(arbb::abs(a), level);
	}

	// sqrsum

	template<typename T>
	inline T sqrsum(const arbb::dense<T, 1>& a)
	{
		return arbb::add_reduce(a * a);
	}

	template<typename T, size_t D>
	inline T sqrsum(const arbb::dense<T, D>& a)
	{
		return sqrsum(a.flatten());
	}

	template<typename T>
	inline arbb::dense<T, 1> sqrsum(const arbb::dense<T, 2>& a, unsigned int level)
	{
		return arbb::add_reduce(a * a, level);
	}

	// L2 norm

	template<typename T>
	inline T norm_L2(const arbb::dense<T, 1>& a)
	{
		return arbb::sqrt(sqrsum(a));
	}

	template<typename T, size_t D>
	inline T norm_L2(const arbb::dense<T, D>& a)
	{
		return norm_L2(a.flatten());
	}

	template<typename T>
	inline arbb::dense<T, 1> norm_L2(const arbb::dense<T, 2>& a, unsigned int level)
	{
		return arbb::sqrt(sqrsum(a, level));
	}

	// Linf norm

	template<typename T>
	inline T norm_Linf(const arbb::dense<T, 1>& a)
	{
		return arbb::max_reduce(arbb::abs(a));
	}

	template<typename T, size_t D>
	inline T norm_Linf(const arbb::dense<T, D>& a)
	{
		return norm_Linf(a.flatten());
	}

	template<typename T>
	inline arbb::dense<T, 1> norm_Linf(const arbb::dense<T, 2>& a, unsigned int level)
	{
		return arbb::max_reduce(arbb::abs(a), level);
	}


	/********************************************
	 *
	 *  matrix/vector multiplication
	 *
	 ********************************************/

	template<typename T>
	inline T dot(const arbb::dense<T, 1>& a, const arbb::dense<T, 1>& b)
	{
		return arbb::add_reduce(a * b);
	}

	template<typename T>
	inline T dot(const arbb::dense<T, 2>& a, const arbb::dense<T, 2>& b)
	{
		return dot(a.flatten(), b.flatten());
	}

	template<typename T>
	inline T vdot(const arbb::dense<T, 1>& a, const arbb::dense<T, 1>& b)
	{
		return arbb::add_reduce(a * b);
	}

	template<typename T>
	inline arbb::dense<T, 1> vdot(const arbb::dense<T, 2>& a, const arbb::dense<T, 2>& b, unsigned int level = 0)
	{
		return arbb::add_reduce(a * b, level);
	}


}

#endif
