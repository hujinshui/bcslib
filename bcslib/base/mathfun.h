/**
 * @file math_functors.h
 *
 * The functors for elementary function computation
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATH_FUNCTORS_H
#define BCSLIB_MATH_FUNCTORS_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_math.h>

#include <functional>


namespace bcs
{

	/******************************************************
	 *
	 *  Comparison functors
	 *
	 ******************************************************/

	// max, min, and bounding

	template<typename T>
	struct max_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x, const T& y) const
		{
			return x > y ? x : y;
		}
	};

	template<typename T>
	struct min_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x, const T& y) const
		{
			return x < y ? x : y;
		}
	};

	template<typename T>
	struct ubound_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T ub;
		ubound_fun(const T& b) : ub(b) { }

		T operator() (const T& x) const
		{
			return x < ub ? x : ub;
		}
	};

	template<typename T>
	struct lbound_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T lb;
		lbound_fun(const T& b) : lb(b) { }

		T operator() (const T& x) const
		{
			return x > lb ? x : lb;
		}
	};

	template<typename T>
	struct clamp_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T lb, ub;
		clamp_fun(const T& lb_, const T& ub_) : lb(lb_), ub(ub_) { }

		T operator() (const T& x) const
		{
			return math::clamp(x, lb, ub);
		}
	};


	/******************************************************
	 *
	 *  Abs and Power functions
	 *
	 ******************************************************/

	template<typename T>
	struct abs_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return math::abs(x);
		}
	};

	template<typename T>
	struct copy_sign_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& a, const T& b) const
		{
			return math::copysign(a, b);
		}
	};

	template<typename T>
	struct sqr_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return math::sqr(x);
		}
	};

	template<typename T>
	struct sqrt_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::sqrt(x);
		}
	};

	template<typename T>
	struct cube_fun: public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return math::cube(x);
		}
	};

	template<typename T>
	struct cbrt_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::cbrt(x);
		}
	};

	template<typename T>
	struct rcp_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::rcp(x);
		}
	};

	template<typename T>
	struct rsqrt_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::rsqrt(x);
		}
	};

	template<typename T>
	struct hypot_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x1, const T& x2) const
		{
			return math::hypot(x1, x2);
		}
	};

	template<typename T>
	struct pow_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x, const T& e) const
		{
			return math::pow(x, e);
		}
	};

	template<typename T>
	struct pow_n_fun : public std::binary_function<T, int, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x, int n)
		{
			return math::pow(x, n);
		}
	};


	/******************************************************
	 *
	 *  Exponential and Power functions
	 *
	 ******************************************************/

	template<typename T>
	struct exp_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::exp(x);
		}
	};

	template<typename T>
	struct exp2_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::exp2(x);
		}
	};

	template<typename T>
	struct log_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::log(x);
		}
	};

	template<typename T>
	struct log10_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::log10(x);
		}
	};

	template<typename T>
	struct log2_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::log2(x);
		}
	};

	template<typename T>
	struct expm1_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::expm1(x);
		}
	};

	template<typename T>
	struct log1p_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::log1p(x);
		}
	};


	/******************************************************
	 *
	 *  Rounding functions
	 *
	 ******************************************************/

	template<typename T>
	struct floor_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::floor(x);
		}
	};

	template<typename T>
	struct ceil_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::ceil(x);
		}
	};

	template<typename T>
	struct round_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::round(x);
		}
	};

	template<typename T>
	struct trunc_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::trunc(x);
		}
	};


	/******************************************************
	 *
	 *  Trigonometric functions
	 *
	 ******************************************************/

	template<typename T>
	struct sin_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::sin(x);
		}
	};

	template<typename T>
	struct cos_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::cos(x);
		}
	};

	template<typename T>
	struct tan_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::tan(x);
		}
	};

	template<typename T>
	struct sincos_fun : public std::unary_function<T, std::pair<T, T> >
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		std::pair<T, T> operator() (const T& x) const
		{
			std::pair<T, T> r;
			math::sincos(x, &(r.first), &(r.second));
			return r;
		}
	};

	template<typename T>
	struct asin_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::asin(x);
		}
	};

	template<typename T>
	struct acos_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::acos(x);
		}
	};

	template<typename T>
	struct atan_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::atan(x);
		}
	};

	template<typename T>
	struct atan2_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& y, const T& x) const
		{
			return math::atan2(y, x);
		}
	};


	/******************************************************
	 *
	 *  Hyperbolic functions
	 *
	 ******************************************************/

	template<typename T>
	struct sinh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::sinh(x);
		}
	};

	template<typename T>
	struct cosh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::cosh(x);
		}
	};

	template<typename T>
	struct tanh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::tanh(x);
		}
	};

	template<typename T>
	struct asinh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::asinh(x);
		}
	};

	template<typename T>
	struct acosh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::acosh(x);
		}
	};

	template<typename T>
	struct atanh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::atanh(x);
		}
	};


	/******************************************************
	 *
	 *  Special functions
	 *
	 ******************************************************/

	template<typename T>
	struct erf_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::erf(x);
		}
	};

	template<typename T>
	struct erfc_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::erfc(x);
		}
	};

	template<typename T>
	struct lgamma_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::lgamma(x);
		}
	};

	template<typename T>
	struct tgamma_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::tgamma(x);
		}
	};


	/******************************************************
	 *
	 *  Number Classification functions
	 *
	 ******************************************************/

	template<typename T>
	struct isfinite_fun : public std::unary_function<T, bool>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::isfinite(x);
		}
	};

	template<typename T>
	struct isinf_fun : public std::unary_function<T, bool>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::isinf(x);
		}
	};

	template<typename T>
	struct isnan_fun : public std::unary_function<T, bool>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::isnan(x);
		}
	};

	template<typename T>
	struct signbit_fun : public std::unary_function<T, bool>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return math::signbit(x);
		}
	};
}


#endif 
