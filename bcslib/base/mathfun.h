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

#include <cmath>
#include <type_traits>
#include <functional>


namespace bcs
{
	/******************************************************
	 *
	 *  Additional math functions
	 *  (including workarounds)
	 *
	 ******************************************************/

	template<typename T>
	BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	sqr(T x)
	{
		return x * x;
	}

	template<typename T>
	BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	cube(T x)
	{
		return x * x * x;
	}

	template<typename T>
	BCS_FORCE_INLINE T rgn_bound(T x, T lb, T ub)
	{
		return x < lb ? lb : (x > ub ? ub : x);
	}

#if BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
	using std::round;
	using std::hypot;
#else
	template<typename T>
	BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	hypot(T x, T y)
	{
		T ax = std::abs(x);
		T ay = std::abs(y);

		T amin, amax;
		if (ax < ay)
		{
			amin = ax;
			amax = ay;
		}
		else
		{
			amin = ay;
			amax = ax;
		}

		return amax > 0 ? amax * std::sqrt(1 + sqr(amin / amax)) : 0;
	}

	template<typename T>
	BCS_FORCE_INLINE typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	round(T x)
	{
		return std::floor(x + (T)(0.5));
	}
#endif


	/******************************************************
	 *
	 *  Functors
	 *
	 ******************************************************/

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
	struct rcp_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return T(1) / x;
		}
	};

	template<typename T>
	struct rsqrt_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_arithmetic<T>);

		T operator() (const T& x) const
		{
			return T(1) / std::sqrt(x);
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
	struct floor_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return std::floor(x);
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
	struct round_fun : public std::unary_function<T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x) const
		{
			return round(x);
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

	template<typename T>
	struct hypot_fun : public std::binary_function<T, T, T>
	{
		BCS_STATIC_ASSERT_V(std::is_floating_point<T>);

		T operator() (const T& x1, const T& x2) const
		{
			return hypot(x1, x2);
		}
	};

}


#endif 
