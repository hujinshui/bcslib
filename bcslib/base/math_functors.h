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

#include <type_traits>
#include <functional>
#include <cmath>

namespace bcs
{

	// max, min, and bounding

	template<typename T>
	struct max_fun : public std::binary_function<T, T, T>
	{
		T operator() (const T& x, const T& y) const
		{
			return x > y ? x : y;
		}
	};

	template<typename T>
	struct min_fun : public std::binary_function<T, T, T>
	{
		T operator() (const T& x, const T& y) const
		{
			return x < y ? x : y;
		}
	};

	template<typename T>
	struct ubound_fun : public std::unary_function<T, T>
	{
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
		T lb;
		lbound_fun(const T& b) : lb(b) { }

		T operator() (const T& x) const
		{
			return x > lb ? x : lb;
		}
	};

	template<typename T>
	struct rgn_bound_fun : public std::unary_function<T, T>
	{
		T lb, ub;
		rgn_bound_fun(const T& lb_, const T& ub_) : lb(lb_), ub(ub_) { }

		T operator() (const T& x) const
		{
			return rgn_bound(x, lb, ub);
		}
	};



	// calculation functors


	template<typename T>
	struct abs_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return std::abs(x);
		}
	};


	template<typename T>
	struct sqr_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return sqr(x);
		}
	};

	template<typename T>
	struct cube_fun: public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return cube(x);
		}
	};

	template<typename T>
	struct sqrt_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::sqrt(x);
		}
	};

	template<typename T>
	struct pow_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x, const T& e) const
		{
			return std::pow(x, e);
		}
	};

	template<typename T>
	struct pow_n_fun : public std::binary_function<T, int, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x, int n)
		{
			return std::pow(x, n);
		}
	};

	template<typename T>
	struct exp_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::exp(x);
		}
	};

	template<typename T>
	struct log_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::log(x);
		}
	};

	template<typename T>
	struct log10_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::log10(x);
		}
	};

	template<typename T>
	struct ceil_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::ceil(x);
		}
	};


	template<typename T>
	struct floor_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::floor(x);
		}
	};


	template<typename T>
	struct sin_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::sin(x);
		}
	};

	template<typename T>
	struct cos_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::cos(x);
		}
	};

	template<typename T>
	struct tan_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::tan(x);
		}
	};

	template<typename T>
	struct asin_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::asin(x);
		}
	};

	template<typename T>
	struct acos_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::acos(x);
		}
	};

	template<typename T>
	struct atan_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::atan(x);
		}
	};

	template<typename T>
	struct atan2_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& y, const T& x) const
		{
			return std::atan2(y, x);
		}
	};

	template<typename T>
	struct sinh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::sinh(x);
		}
	};

	template<typename T>
	struct cosh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::cosh(x);
		}
	};

	template<typename T>
	struct tanh_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::tanh(x);
		}
	};



}


#endif 
