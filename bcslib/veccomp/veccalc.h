/**
 * @file veccalc.h
 *
 * Vectorized calculation
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECCALC_H
#define BCSLIB_VECCALC_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>

namespace bcs
{

	// generic calculation functions

	template<typename T, typename TFunc>
	void vec_calc(size_t n, const T *x, T *y, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*y++ = f(*x++);
		}
	}

	template<typename T, typename TFunc>
	void vec_calc(size_t n, const T *x1, const T *x2, T *y, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*y++ = f(*x1++, *x2++);
		}
	}

	template<typename T, typename TFunc>
	void vec_calc(size_t n, const T *x1, const T& x2, T *y, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*y++ = f(*x1++, x2);
		}
	}

	template<typename T, typename TFunc>
	void vec_calc(size_t n, const T& x1, const T* x2, T *y, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*y++ = f(x1, *x2++);
		}
	}

	template<typename T, typename TFunc>
	void vec_calc_inplace(size_t n, T *y, const T *x, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			f(*y++, *x++);
		}
	}

	template<typename T, typename TFunc>
	void vec_calc_inplace(size_t n, T *y, const T &x, TFunc f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			f(*y++, x);
		}
	}


	// simple calculation: add, subtract, multiple, divide, and negate

	// add

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T *x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::plus<T>());
	}

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T& x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::plus<T>());
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T *x)
	{
		vec_calc_inplace(n, y, x, inplace_plus<T>());
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T& x)
	{
		vec_calc_inplace(n, y, x, inplace_plus<T>());
	}

	// sub

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T *x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::minus<T>());
	}

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T& x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::minus<T>());
	}

	template<typename T>
	inline void vec_sub(size_t n, const T& x1, const T* x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::minus<T>());
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T *x)
	{
		vec_calc_inplace(n, y, x, inplace_minus<T>());
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T& x)
	{
		vec_calc_inplace(n, y, x, inplace_minus<T>());
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, const T& x, T *y)
	{
		vec_sub(n, x, y, y);
	}


	// mul

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T *x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::multiplies<T>());
	}

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T& x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::multiplies<T>());
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T *x)
	{
		vec_calc_inplace(n, y, x, inplace_multiplies<T>());
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T& x)
	{
		vec_calc_inplace(n, y, x, inplace_multiplies<T>());
	}

	// div

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T *x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::divides<T>());
	}

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T& x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::divides<T>());
	}

	template<typename T>
	inline void vec_div(size_t n, const T& x1, const T* x2, T *y)
	{
		vec_calc(n, x1, x2, y, std::divides<T>());
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T *x)
	{
		vec_calc_inplace(n, y, x, inplace_divides<T>());
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T& x)
	{
		vec_calc_inplace(n, y, x, inplace_divides<T>());
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, const T& x, T *y)
	{
		vec_div(n, x, y, y);
	}

	// negate

	template<typename T>
	inline void vec_negate(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, std::negate<T>());
	}

	template<typename T>
	inline void vec_negate(size_t n, T *y)
	{
		vec_negate(n, y, y);
	}


	// elementary functions

	template<typename T>
	inline void vec_abs(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, abs_fun<T>());
	}

	template<typename T>
	inline void vec_sqr(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, sqr_fun<T>());
	}

	template<typename T>
	inline void vec_sqrt(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, sqrt_fun<T>());
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T* e, T *y)
	{
		vec_calc(n, x, e, y, pow_fun<T>());
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T& e, T *y)
	{
		vec_calc(n, x, e, y, abs_fun<T>());
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, int e, T *y)
	{
		vec_calc(n, x, y, pow_n_fun<T>(e));
	}

	template<typename T>
	inline void vec_exp(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, exp_fun<T>());
	}

	template<typename T>
	inline void vec_log(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, log_fun<T>());
	}

	template<typename T>
	inline void vec_log10(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, log10_fun<T>());
	}

	template<typename T>
	inline void vec_ceil(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, ceil_fun<T>());
	}

	template<typename T>
	inline void vec_floor(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, floor_fun<T>());
	}

	template<typename T>
	inline void vec_sin(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, sin_fun<T>());
	}

	template<typename T>
	inline void vec_cos(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, cos_fun<T>());
	}

	template<typename T>
	inline void vec_tan(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, tan_fun<T>());
	}

	template<typename T>
	inline void vec_asin(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, asin_fun<T>());
	}

	template<typename T>
	inline void vec_acos(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, acos_fun<T>());
	}

	template<typename T>
	inline void vec_atan(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, atan_fun<T>());
	}

	template<typename T>
	inline void vec_atan2(size_t n, const T *x1, const T* x2, T *y)
	{
		vec_calc(n, x1, x2, y, atan2_fun<T>());
	}

	template<typename T>
	inline void vec_sinh(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, sinh_fun<T>());
	}

	template<typename T>
	inline void vec_cosh(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, cosh_fun<T>());
	}

	template<typename T>
	inline void vec_tanh(size_t n, const T *x, T *y)
	{
		vec_calc(n, x, y, tanh_fun<T>());
	}
}

#endif 
