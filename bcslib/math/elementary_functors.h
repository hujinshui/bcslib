/**
 * @file elementary_functors.h
 *
 * Functors for elementary function
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif


#ifndef BCSLIB_ELEMENTARY_FUNCTORS_H_
#define BCSLIB_ELEMENTARY_FUNCTORS_H_

#include <bcslib/core/functor_base.h>

namespace bcs
{
	template<typename T>
	struct unary_sqrt
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::sqrt(x);
		}
	};

	template<typename T>
	struct unary_pow
	{
		typedef T result_type;

		T exponent;

		BCS_ENSURE_INLINE unary_pow(T e) : exponent(e) { }

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::pow(x, exponent);
		}
	};


	template<typename T>
	struct unary_exp
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::exp(x);
		}
	};

	template<typename T>
	struct unary_log
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::log(x);
		}
	};

	template<typename T>
	struct unary_log10
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::log10(x);
		}
	};


	template<typename T>
	struct unary_floor
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::floor(x);
		}
	};


	template<typename T>
	struct unary_ceil
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::ceil(x);
		}
	};


	template<typename T>
	struct unary_sin
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::sin(x);
		}
	};


	template<typename T>
	struct unary_cos
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::cos(x);
		}
	};


	template<typename T>
	struct unary_tan
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::tan(x);
		}
	};


	template<typename T>
	struct unary_asin
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::asin(x);
		}
	};


	template<typename T>
	struct unary_acos
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::acos(x);
		}
	};


	template<typename T>
	struct unary_atan
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::atan(x);
		}
	};


	template<typename T>
	struct binary_atan2
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x, const T& y) const
		{
			return bcs::math::atan2(x, y);
		}
	};


	template<typename T>
	struct unary_sinh
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::sinh(x);
		}
	};


	template<typename T>
	struct unary_cosh
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::cosh(x);
		}
	};


	template<typename T>
	struct unary_tanh
	{
		typedef T result_type;

		BCS_ENSURE_INLINE T operator() (const T& x) const
		{
			return bcs::math::tanh(x);
		}
	};


	DECLARE_UNARY_EWISE_FUNCTOR( unary_sqrt )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_pow )

	DECLARE_UNARY_EWISE_FUNCTOR( unary_exp )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_log )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_log10 )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_floor )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_ceil )

	DECLARE_UNARY_EWISE_FUNCTOR( unary_sin )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_cos )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_tan )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_asin )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_acos )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_atan )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_sinh )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_cosh )
	DECLARE_UNARY_EWISE_FUNCTOR( unary_tanh )

	DECLARE_BINARY_EWISE_FUNCTOR( binary_atan2 )
}

#endif


