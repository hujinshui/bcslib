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
#include <bcslib/base/mathfun.h>

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

	// max_each and min_each

	template<typename T>
	inline void vec_max_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x1[i], x2[i]);
	}

	template<typename T>
	inline void vec_min_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x1[i], x2[i]);
	}


	/********************************************
	 *
	 *  Bounding (Thresholding)
	 *
	 *******************************************/

	template<typename T>
	inline void vec_lbound(size_t n, const T* x, const T& lb, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x[i], lb);
	}

	template<typename T>
	inline void vec_lbound_inplace(size_t n, T* y, const T& lb)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (y[i] < lb) y[i] = lb;
		}
	}

	template<typename T>
	inline void vec_ubound(size_t n, const T* x, const T& ub, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x[i], ub);
	}

	template<typename T>
	inline void vec_ubound_inplace(size_t n, T* y, const T& ub)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (y[i] > ub) y[i] = ub;
		}
	}

	template<typename T>
	inline void vec_rgn_bound(size_t n, const T* x, const T& lb, const T& ub, T *y)
	{
		for (size_t i = 0; i < n; ++i)
		{
			y[i] = math::rgn_bound(x[i], lb, ub);
		}
	}

	template<typename T>
	inline void vec_rgn_bound_inplace(size_t n, T* y, const T& lb, const T& ub)
	{
		for (size_t i = 0; i < n; ++i)
		{
			T v = y[i];
			if (v < lb) y[i] = lb;
			else if (v > ub) y[i] = ub;
		}
	}

	template<typename T>
	inline void vec_abound(size_t n, const T* x, const T& ab, T *y)
	{
		vec_rgn_bound(n, x, -ab, ab, y);
	}

	template<typename T>
	inline void vec_abound_inplace(size_t n, T* y, const T& ab)
	{
		vec_rgn_bound_inplace(n, y, -ab, ab);
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


	/********************************************
	 *
	 *  Elementary functions
	 *
	 *******************************************/

	// abs and power functions

	template<typename T>
	inline void vec_abs(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::abs(x[i]);
	}

	template<typename T>
	inline void vec_copysign(size_t n, const T *x, T *s, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::copysign(x[i], s[i]);
	}

	template<typename T>
	inline void vec_sqr(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::sqr(x[i]);
	}

	template<typename T>
	inline void vec_sqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::sqrt(x[i]);
	}

	template<typename T>
	inline void vec_cube(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::cube(x[i]);
	}

	template<typename T>
	inline void vec_cbrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::cbrt(x[i]);
	}

	template<typename T>
	inline void vec_rcp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::rcp(x[i]);
	}

	template<typename T>
	inline void vec_rsqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::rsqrt(x[i]);
	}

	template<typename T>
	inline void vec_hypot(size_t n, const T *x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::hypot(x1[i], x2[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T* e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::pow(x[i], e[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T& e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::pow(x[i], e);
	}


	// exponential and logarithm functions

	template<typename T>
	inline void vec_exp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::exp(x[i]);
	}

	template<typename T>
	inline void vec_exp2(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::exp2(x[i]);
	}

	template<typename T>
	inline void vec_log(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::log(x[i]);
	}

	template<typename T>
	inline void vec_log10(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::log10(x[i]);
	}

	template<typename T>
	inline void vec_log2(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::log2(x[i]);
	}

	template<typename T>
	inline void vec_expm1(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::expm1(x[i]);
	}

	template<typename T>
	inline void vec_log1p(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::log1p(x[i]);
	}


	// rounding functions

	template<typename T>
	inline void vec_floor(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::floor(x[i]);
	}

	template<typename T>
	inline void vec_ceil(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::ceil(x[i]);
	}

	template<typename T>
	inline void vec_round(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::round(x[i]);
	}

	template<typename T>
	inline void vec_trunc(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::trunc(x[i]);
	}


	// trigonometric functions

	template<typename T>
	inline void vec_sin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::sin(x[i]);
	}

	template<typename T>
	inline void vec_cos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::cos(x[i]);
	}

	template<typename T>
	inline void vec_tan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::tan(x[i]);
	}

	template<typename T>
	inline void vec_sincos(size_t n, const T *x, T *s, T *c)
	{
		for (size_t i = 0; i < n; ++i) math::sincos(x[i], s[i], c[i]);
	}

	template<typename T>
	inline void vec_asin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::asin(x[i]);
	}

	template<typename T>
	inline void vec_acos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::acos(x[i]);
	}

	template<typename T>
	inline void vec_atan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::atan(x[i]);
	}

	template<typename T>
	inline void vec_atan2(size_t n, const T *x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::atan2(x1[i], x2[i]);
	}


	// hyperbolic functions

	template<typename T>
	inline void vec_sinh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::sinh(x[i]);
	}

	template<typename T>
	inline void vec_cosh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::cosh(x[i]);
	}

	template<typename T>
	inline void vec_tanh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::tanh(x[i]);
	}

	template<typename T>
	inline void vec_asinh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::asinh(x[i]);
	}

	template<typename T>
	inline void vec_acosh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::acosh(x[i]);
	}

	template<typename T>
	inline void vec_atanh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::atanh(x[i]);
	}


	// special functions

	template<typename T>
	inline void vec_erf(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::erf(x[i]);
	}

	template<typename T>
	inline void vec_erfc(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::erfc(x[i]);
	}

	template<typename T>
	inline void vec_lgamma(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::lgamma(x[i]);
	}

	template<typename T>
	inline void vec_tgamma(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::tgamma(x[i]);
	}


	// classification functions

	template<typename T>
	inline void vec_isfinite(size_t n, const T *x, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::isfinite(x[i]);
	}

	template<typename T>
	inline void vec_isinf(size_t n, const T *x, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::isinf(x[i]);
	}

	template<typename T>
	inline void vec_isnan(size_t n, const T *x, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::isnan(x[i]);
	}

	template<typename T>
	inline void vec_signbit(size_t n, const T *x, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = math::signbit(x[i]);
	}
}



/********************************************
 *
 *  Vendor-specific implementations
 *
 *******************************************/

#ifdef BCS_ENABLE_INTEL_IPPS
#include <bcslib/veccomp/intel_calc.h>
#endif


#endif


