/**
 * @file veccalc.h
 *
 * Vectorized calculation
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECCALC_H
#define BCSLIB_VECCALC_H

#include <bcslib/base/basic_defs.h>

#include <cmath>
#include <algorithm>

namespace bcs
{
	/********************************************
	 *
	 *  Order comparison
	 *
	 *******************************************/

	// eq

	template<typename T>
	inline void vec_eq(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] == x2[i]);
	}

	template<typename T>
	inline void vec_eq(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] == x2);
	}

	// ne

	template<typename T>
	inline void vec_ne(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] != x2[i]);
	}

	template<typename T>
	inline void vec_ne(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] != x2);
	}

	// gt

	template<typename T>
	inline void vec_gt(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] > x2[i]);
	}

	template<typename T>
	inline void vec_gt(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] > x2);
	}

	// ge

	template<typename T>
	inline void vec_ge(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] >= x2[i]);
	}

	template<typename T>
	inline void vec_ge(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] >= x2);
	}

	// lt

	template<typename T>
	inline void vec_lt(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] < x2[i]);
	}

	template<typename T>
	inline void vec_lt(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] < x2);
	}

	// le

	template<typename T>
	inline void vec_le(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] <= x2[i]);
	}

	template<typename T>
	inline void vec_le(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] <= x2);
	}

	// max_each

	template<typename T>
	inline void vec_max_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x1[i], x2[i]);
	}

	template<typename T>
	inline void vec_max_each(size_t n, const T* x1, const T& x2, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x1[i], x2);
	}

	// min_each

	template<typename T>
	inline void vec_min_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x1[i], x2[i]);
	}

	template<typename T>
	inline void vec_min_each(size_t n, const T* x1, const T& x2, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x1[i], x2);
	}



	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// add

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] + x2[i];
	}

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] + x2;
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] += x[i];
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] += x;
	}


	// sub

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] - x2[i];
	}

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] - x2;
	}

	template<typename T>
	inline void vec_sub(size_t n, const T& x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1 - x2[i];
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] -= x[i];
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] -= x;
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, const T& x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x - y[i];
	}


	// mul

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] * x2[i];
	}

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] * x2;
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] *= x[i];
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] *= x;
	}

	// div

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] / x2[i];
	}

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] / x2;
	}

	template<typename T>
	inline void vec_div(size_t n, const T& x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1 / x2[i];
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] /= x[i];
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] /= x;
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, const T& x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x / y[i];
	}

	// negate

	template<typename T>
	inline void vec_negate(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = -x[i];
	}

	template<typename T>
	inline void vec_negate(size_t n, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = -y[i];
	}


	// absolute values

	template<typename T>
	inline void vec_abs(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::abs(x[i]);
	}


	/********************************************
	 *
	 *  Elementary Function Evaluation
	 *
	 *******************************************/

	// power and root functions

	template<typename T>
	inline void vec_sqr(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = sqr(x[i]);
	}

	template<typename T>
	inline void vec_sqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sqrt(x[i]);
	}

	template<typename T>
	inline void vec_rcp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = T(1) / x[i];
	}

	template<typename T>
	inline void vec_rsqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = T(1) / std::sqrt(x[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T* e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T& e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e);
	}


	// exponential and logarithm functions

	template<typename T>
	inline void vec_exp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::exp(x[i]);
	}

	template<typename T>
	inline void vec_log(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::log(x[i]);
	}

	template<typename T>
	inline void vec_log10(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::log10(x[i]);
	}


	// rounding functions

	template<typename T>
	inline void vec_ceil(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::ceil(x[i]);
	}

	template<typename T>
	inline void vec_floor(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::floor(x[i]);
	}


	// trigonometric functions

	template<typename T>
	inline void vec_sin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sin(x[i]);
	}

	template<typename T>
	inline void vec_cos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::cos(x[i]);
	}

	template<typename T>
	inline void vec_tan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::tan(x[i]);
	}

	template<typename T>
	inline void vec_asin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::asin(x[i]);
	}

	template<typename T>
	inline void vec_acos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::acos(x[i]);
	}

	template<typename T>
	inline void vec_atan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::atan(x[i]);
	}

	template<typename T>
	inline void vec_atan2(size_t n, const T *x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::atan2(x1[i], x2[i]);
	}


	// hyperbolic functions

	template<typename T>
	inline void vec_sinh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sinh(x[i]);
	}

	template<typename T>
	inline void vec_cosh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::cosh(x[i]);
	}

	template<typename T>
	inline void vec_tanh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
	}
}

#endif 
