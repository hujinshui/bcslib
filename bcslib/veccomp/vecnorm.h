/**
 * @file vecnorm.h
 *
 * Computation of vector norms
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECNORM_H
#define BCSLIB_VECNORM_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>

namespace bcs
{
	template<typename T>
	inline T vec_dot_prod(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += x[i] * y[i];
		}
		return s;
	}

	template<typename T>
	inline T vec_xlogy_sum(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i] != 0)
			{
				s += x[i] * log(y[i]);
 			}
		}
		return s;
	}


	template<typename T>
	inline size_t vec_nonzeros_count(size_t n, const T *x)
	{
		size_t c = 0;
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i] != 0) ++c;
		}
		return c;
	}


	template<typename T>
	inline T vec_abs_sum(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i) s += std::abs(x[i]);
		return s;
	}

	template<typename T>
	inline T vec_sqr_sum(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i) s += sqr(x[i]);
		return s;
	}

	template<typename T>
	inline T vec_abs_max(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			T a = std::abs(x[i]);
			if (a > s) s = a;
		}

		return s;
	}


	template<typename T, typename TPower>
	inline T vec_pow_sum(size_t n, const T *x, TPower p)
	{
		if (p == 2)
		{
			return vec_sqr_sum(n, x);
		}
		else if (p == 1)
		{
			return vec_abs_sum(n, x);
		}
		else if (p == 0)
		{
			return vec_nonzeros_count(n, x);
		}
		else
		{
			T s(0);
			for (size_t i = 0; i < n; ++i)
			{
				s += std::pow(x[i], p);
			}
			return s;
		}
	}


	template<typename T>
	inline T vec_L1norm(size_t n, const T *x)
	{
		return vec_abs_sum(n, x);
	}

	template<typename T>
	inline T vec_L2norm(size_t n, const T *x)
	{
		return std::sqrt(vec_sqr_sum(n, x));
	}


	template<typename T>
	inline T vec_Linfnorm(size_t n, const T *x)
	{
		return vec_abs_max(n, x);
	}

	template<typename T, typename TPower>
	inline T vec_Lpnorm(size_t n, const T *x, TPower p)
	{
		if (p < 1)
		{
			throw invalid_argument("The p-value of a norm should be no less than 1.");
		}

		if (p == 2)
		{
			return vec_L2norm(n, x);
		}
		else if (p == 1)
		{
			return vec_L1norm(n, x);
		}
		else
		{
			return (T)std::pow(vec_pow_sum(n, x, p), 1.0 / p);
		}
	}

}

#endif 
