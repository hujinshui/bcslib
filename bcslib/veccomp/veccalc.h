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

namespace bcs
{

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
		for (size_t i = 0; i < n; ++i) y[i] = x;
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


	/********************************************
	 *
	 *  Arithmetic Functors
	 *
	 *******************************************/

	template<typename T>
	struct vec_add_functor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const { return vec_add(n, x1, x2, y); }
		void operator() (size_t n, const T *x1, const T &x2, T *y) const { return vec_add(n, x1, x2, y); }
		void operator() (size_t n, const T &x1, const T *x2, T *y) const { return vec_add(n, x2, x1, y); }

		void operator() (size_t n, T *y, const T *x) const { return vec_add_inplace(n, y, x); }
		void operator() (size_t n, T *y, const T &x) const { return vec_add_inplace(n, y, x); }
	};


	/********************************************
	 *
	 *  Elementary Function Evaluation
	 *
	 *******************************************/

	template<typename T>
	inline void vec_abs(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::abs(x[i]);
	}

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
	inline void vec_pow(size_t n, const T *x, const T* e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T& e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e);
	}

	template<typename T>
	inline void vec_pow_n(size_t n, const T *x, int e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e);
	}

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
		for (size_t i = 0; i < n; ++i) y[i] = std::tan(x1[i], x2[i]);
	}

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
