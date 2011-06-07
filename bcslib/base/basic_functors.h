/**
 * @file basic_funcs.h
 *
 * Definitions of some basic function objects
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BASIC_FUNCTORS_H
#define BCSLIB_BASIC_FUNCTORS_H

#include <bcslib/base/config.h>
#include <functional>

namespace bcs
{

	// inplace function adaptors

	template<typename T>
	struct inplace
	{
		std::function<T(const T&)> func;

		template<typename F>
		inplace(F f) : func(f) { }

		void operator() (T& y) const
		{
			y = f(y);
		}
	};


	// inplace functors

	template<typename T>
	struct inplace_plus
	{
		void operator() (T& y, const T& x) const
		{
			y += x;
		}
	};

	template<typename T>
	struct inplace_minus
	{
		void operator() (T& y, const T& x) const
		{
			y -= x;
		}
	};


	template<typename T>
	struct inplace_multiplies
	{
		void operator() (T& y, const T& x) const
		{
			y *= x;
		}
	};


	template<typename T>
	struct inplace_divides
	{
		void operator() (T& y, const T& x) const
		{
			y /= x;
		}
	};

	template<typename T>
	struct inplace_negate
	{
		void operator()(T& y) const
		{
			y = -y;
		}
	};

}

#endif
