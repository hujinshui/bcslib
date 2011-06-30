/**
 * @file veccalc_functors.h
 *
 * The functors for vectorized calculation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#include <bcslib/veccomp/veccalc.h>

#ifndef BCSLIB_VECCALC_FUNCTORS_H_
#define BCSLIB_VECCALC_FUNCTORS_H_

namespace bcs
{
	/********************************************
	 *
	 *  Order comparison
	 *
	 *******************************************/

	// equal to

	template<typename T>
	struct vec_vec_eq_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_eq(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_eq_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_eq_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_eq(n, x1, r, y);
		}
	};


	// not equal

	template<typename T>
	struct vec_vec_ne_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_ne(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_ne_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_ne_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ne(n, x1, r, y);
		}
	};

	// greater than

	template<typename T>
	struct vec_vec_gt_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_gt(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_gt_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_gt_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_gt(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_gt_ftor
	{
		typedef bool result_value_type;

		T r;
		sca_vec_gt_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_lt(n, x1, r, y);
		}
	};


	// greater than or equal to

	template<typename T>
	struct vec_vec_ge_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_ge(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_ge_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_ge_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ge(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_ge_ftor
	{
		typedef bool result_value_type;

		T r;
		sca_vec_ge_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_le(n, x1, r, y);
		}
	};


	// less than

	template<typename T>
	struct vec_vec_lt_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_lt(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_lt_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_lt_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_lt(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_lt_ftor
	{
		typedef bool result_value_type;

		T r;
		sca_vec_lt_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_gt(n, x1, r, y);
		}
	};


	// less than or equal to

	template<typename T>
	struct vec_vec_le_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_le(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_le_ftor
	{
		typedef bool result_value_type;

		T r;
		vec_sca_le_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_le(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_le_ftor
	{
		typedef bool result_value_type;

		T r;
		sca_vec_le_ftor(const T& r_)
		: r(r_) { }

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ge(n, x1, r, y);
		}
	};


	// max_each

	template<typename T>
	struct vec_vec_max_each_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_max_each(n, x1, x2, y);
		}
	};


	// min_each

	template<typename T>
	struct vec_vec_min_each_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_min_each(n, x1, x2, y);
		}
	};


	/********************************************
	 *
	 *  Bounding
	 *
	 *******************************************/

	template<typename T>
	struct vec_lbound_ftor
	{
		typedef T result_value_type;

		T b;
		vec_lbound_ftor(const T& b_) : b(b_) { }

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_lbound(n, x, b, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_lbound_inplace(n, y, b);
		}
	};


	template<typename T>
	struct vec_ubound_ftor
	{
		typedef T result_value_type;

		T b;
		vec_ubound_ftor(const T& b_) : b(b_) { }

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_ubound(n, x, b, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_ubound_inplace(n, y, b);
		}
	};


	template<typename T>
	struct vec_rgn_bound_ftor
	{
		typedef T result_value_type;

		T lb, ub;
		vec_rgn_bound_ftor(const T& lb_, const T& ub_) : lb(lb_), ub(ub_) { }

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_rgn_bound(n, x, lb, ub, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_rgn_bound_inplace(n, y, lb, ub);
		}
	};


	template<typename T>
	struct vec_abound_ftor
	{
		typedef T result_value_type;

		T ab;
		vec_abound_ftor(const T& ab_) : ab(ab_) { }

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_abound(n, x, ab, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_abound_inplace(n, y, ab);
		}
	};


	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// addition

	template<typename T>
	struct vec_vec_add_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_add(n, x1, x2, y);
		}

		void operator() (size_t n, T*y, const T*x1) const
		{
			vec_add_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_add_ftor
	{
		typedef T result_value_type;

		T s;
		vec_sca_add_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_add(n, x1, s, y);
		}

		void operator() (size_t n, T* y) const
		{
			return vec_add_inplace(n, y, s);
		}
	};


	// subtraction

	template<typename T>
	struct vec_vec_sub_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_sub(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T *x1) const
		{
			vec_sub_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_sub_ftor
	{
		typedef T result_value_type;

		T s;
		vec_sca_sub_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_sub(n, x1, s, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_sub_inplace(n, y, s);
		}
	};

	template<typename T>
	struct sca_vec_sub_ftor
	{
		typedef T result_value_type;

		T s;
		sca_vec_sub_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x2, T *y) const
		{
			vec_sub(n, s, x2, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_sub_inplace(n, s, y);
		}
	};


	// multiplication

	template<typename T>
	struct vec_vec_mul_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_mul(n, x1, x2, y);
		}

		void operator() (size_t n, T*y, const T*x1) const
		{
			vec_mul_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_mul_ftor
	{
		typedef T result_value_type;

		T s;
		vec_sca_mul_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_mul(n, x1, s, y);
		}

		void operator() (size_t n, T* y) const
		{
			return vec_mul_inplace(n, y, s);
		}
	};


	// division

	template<typename T>
	struct vec_vec_div_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_div(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T *x1) const
		{
			vec_div_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_div_ftor
	{
		typedef T result_value_type;

		T s;
		vec_sca_div_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_div(n, x1, s, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_div_inplace(n, y, s);
		}
	};

	template<typename T>
	struct sca_vec_div_ftor
	{
		typedef T result_value_type;

		T s;
		sca_vec_div_ftor(const T& s_)
		: s(s_) { }

		void operator() (size_t n, const T *x2, T *y) const
		{
			vec_div(n, s, x2, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_div_inplace(n, s, y);
		}
	};



	// negation

	template<typename T>
	struct vec_neg_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_negate(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_negate(n, y);
		}
	};


	/********************************************
	 *
	 *  Absolute value and Power functions
	 *
	 *******************************************/

	// abs

	template<typename T>
	struct vec_abs_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_abs(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_abs(n, y, y);
		}
	};

	// copysign

	template<typename T>
	struct vec_copysign_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, const T *s, T *y) const
		{
			return vec_copysign(n, x, s, y);
		}

		void operator() (size_t n, T *y, const T *s) const
		{
			return vec_copysign(n, y, s, y);
		}
	};

	// sqr

	template<typename T>
	struct vec_sqr_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sqr(n, x, y);
		}
	};


	// sqrt

	template<typename T>
	struct vec_sqrt_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sqrt(n, x, y);
		}
	};


	// cube

	template<typename T>
	struct vec_cube_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_cube(n, x, y);
		}
	};


	// sqrt

	template<typename T>
	struct vec_cbrt_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sqrt(n, x, y);
		}
	};


	// rcp

	template<typename T>
	struct vec_rcp_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_rcp(n, x, y);
		}
	};


	// rsqrt

	template<typename T>
	struct vec_rsqrt_ftor : public std::unary_function<T, T>
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_rsqrt(n, x, y);
		}
	};

	template<typename T>
	struct vec_hypot_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			return vec_hypot(n, x1, x2, y);
		}
	};


	// pow (vec-vec)

	template<typename T>
	struct vec_pow_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, const T *e, T *y) const
		{
			return vec_pow(n, x, e, y);
		}
	};


	// pow (vec-scalar)

	template<typename T>
	struct vec_sca_pow_ftor
	{
		typedef T result_value_type;

		T e;
		vec_sca_pow_ftor(const T& e_)
		: e(e_) { }

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_pow(n, x, e, y);
		}
	};



	/********************************************
	 *
	 *  Exponential and Logarithm functions
	 *
	 *******************************************/

	// exp

	template<typename T>
	struct vec_exp_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_exp(n, x, y);
		}
	};

	// exp2

	template<typename T>
	struct vec_exp2_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_exp2(n, x, y);
		}
	};

	// log

	template<typename T>
	struct vec_log_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log(n, x, y);
		}
	};

	// log10

	template<typename T>
	struct vec_log10_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log10(n, x, y);
		}
	};

	// log2

	template<typename T>
	struct vec_log2_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log2(n, x, y);
		}
	};

	// expm1

	template<typename T>
	struct vec_expm1_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_expm1(n, x, y);
		}
	};

	// log1p

	template<typename T>
	struct vec_log1p_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log1p(n, x, y);
		}
	};


	/********************************************
	 *
	 *  Rounding functions
	 *
	 *******************************************/

	// floor

	template<typename T>
	struct vec_floor_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_floor(n, x, y);
		}
	};

	// ceil

	template<typename T>
	struct vec_ceil_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_ceil(n, x, y);
		}
	};

	// round

	template<typename T>
	struct vec_round_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_round(n, x, y);
		}
	};

	// trunc

	template<typename T>
	struct vec_trunc_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_trunc(n, x, y);
		}
	};


	/********************************************
	 *
	 *  Trigonometric functions
	 *
	 *******************************************/

	// sin

	template<typename T>
	struct vec_sin_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sin(n, x, y);
		}
	};

	// cos

	template<typename T>
	struct vec_cos_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_cos(n, x, y);
		}
	};

	// tan

	template<typename T>
	struct vec_tan_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_tan(n, x, y);
		}
	};

	// asin

	template<typename T>
	struct vec_asin_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_asin(n, x, y);
		}
	};

	// acos

	template<typename T>
	struct vec_acos_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_acos(n, x, y);
		}
	};

	// atan

	template<typename T>
	struct vec_atan_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_atan(n, x, y);
		}
	};

	// atan2

	template<typename T>
	struct vec_atan2_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x1, const T* x2, T *y) const
		{
			return vec_atan2(n, x1, x2, y);
		}
	};


	/********************************************
	 *
	 *  Hyperbolic functions
	 *
	 *******************************************/

	// sinh

	template<typename T>
	struct vec_sinh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sinh(n, x, y);
		}
	};

	// cosh

	template<typename T>
	struct vec_cosh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_cosh(n, x, y);
		}
	};

	// tanh

	template<typename T>
	struct vec_tanh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_tanh(n, x, y);
		}
	};

	// asinh

	template<typename T>
	struct vec_asinh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_asinh(n, x, y);
		}
	};

	// acosh

	template<typename T>
	struct avec_cosh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_acosh(n, x, y);
		}
	};

	// atanh

	template<typename T>
	struct vec_atanh_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_atanh(n, x, y);
		}
	};


	/********************************************
	 *
	 *  Special functions
	 *
	 *******************************************/

	// erf

	template<typename T>
	struct vec_erf_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_erf(n, x, y);
		}
	};

	// erfc

	template<typename T>
	struct vec_erfc_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_erfc(n, x, y);
		}
	};

	// lgamma

	template<typename T>
	struct vec_lgamma_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_lgamma(n, x, y);
		}
	};

	// tgamma

	template<typename T>
	struct vec_tgamma_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_tgamma(n, x, y);
		}
	};


	/********************************************
	 *
	 *  Classification functions
	 *
	 *******************************************/

	// isfinite

	template<typename T>
	struct vec_isfinite_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x, bool *y) const
		{
			return vec_isfinite(n, x, y);
		}
	};

	// isinf

	template<typename T>
	struct vec_isinf_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x, bool *y) const
		{
			return vec_isinf(n, x, y);
		}
	};

	// isnan

	template<typename T>
	struct vec_isnan_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x, bool *y) const
		{
			return vec_isnan(n, x, y);
		}
	};

	// signbit

	template<typename T>
	struct vec_signbit_ftor
	{
		typedef bool result_value_type;

		void operator() (size_t n, const T *x, bool *y) const
		{
			return vec_signbit(n, x, y);
		}
	};
}

#endif
