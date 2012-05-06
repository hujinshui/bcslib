/**
 * @file ewise_functors.h
 *
 * Element-wise arithmetic functors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARITHMETIC_FUNCTORS_H_
#define BCSLIB_ARITHMETIC_FUNCTORS_H_

#include <bcslib/core/functional.h>
#include <bcslib/math/scalar_math.h>
#include <algorithm>

namespace bcs
{

	template<typename T>
	struct binary_plus
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return x + y;
		}
	};

	template<typename T>
	struct binary_minus
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return x - y;
		}
	};

	template<typename T>
	struct binary_times
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return x * y;
		}
	};

	template<typename T>
	struct binary_divides
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return x / y;
		}
	};


	template<typename T>
	struct plus_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE plus_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return x + scalar_arg;
		}
	};

	template<typename T>
	struct minus_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE minus_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return x - scalar_arg;
		}
	};

	template<typename T>
	struct rminus_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE rminus_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return scalar_arg - x;
		}
	};

	template<typename T>
	struct times_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE times_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return x * scalar_arg;
		}
	};

	template<typename T>
	struct divides_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE divides_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return x / scalar_arg;
		}
	};

	template<typename T>
	struct rdivides_scalar
	{
		typedef T result_type;
		T scalar_arg;

		BCS_ENSURE_INLINE rdivides_scalar(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return scalar_arg / x;
		}
	};



	template<typename T>
	struct unary_negate
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return -x;
		}
	};

	template<typename T>
	struct unary_rcp
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::rcp(x);
		}
	};

	template<typename T>
	struct unary_abs
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::abs(x);
		}
	};

	template<typename T>
	struct unary_sqr
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::sqr(x);
		}
	};

	template<typename T>
	struct unary_cube
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::cube(x);
		}
	};


	template<typename T>
	struct binary_min
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return std::min(x, y);
		}
	};


	template<typename T>
	struct unary_min
	{
		typedef T result_type;

		T scalar_arg;

		BCS_ENSURE_INLINE unary_min(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return std::min(x, scalar_arg);
		}
	};


	template<typename T>
	struct binary_max
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return std::max(x, y);
		}
	};


	template<typename T>
	struct unary_max
	{
		typedef T result_type;

		T scalar_arg;

		BCS_ENSURE_INLINE unary_max(const T& s) : scalar_arg(s) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return std::max(x, scalar_arg);
		}
	};


	BCS_DECLARE_EWISE_FUNCTOR( binary_plus, 2 )
	BCS_DECLARE_EWISE_FUNCTOR( binary_minus, 2 )
	BCS_DECLARE_EWISE_FUNCTOR( binary_times, 2 )
	BCS_DECLARE_EWISE_FUNCTOR( binary_divides, 2 )

	BCS_DECLARE_EWISE_FUNCTOR( plus_scalar, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( minus_scalar, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( rminus_scalar, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( times_scalar, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( divides_scalar, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( rdivides_scalar, 1 )

	BCS_DECLARE_EWISE_FUNCTOR( unary_negate, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_rcp, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_abs, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_sqr, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_cube, 1 )

	BCS_DECLARE_EWISE_FUNCTOR( binary_min, 2 )
	BCS_DECLARE_EWISE_FUNCTOR( binary_max, 2 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_min, 1 )
	BCS_DECLARE_EWISE_FUNCTOR( unary_max, 1 )

}

#endif /* EWISE_FUNCTORS_H_ */
