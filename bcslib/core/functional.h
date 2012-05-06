/**
 * @file functional.h
 *
 * The support of functional programming
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_FUNCTIONAL_H_
#define BCSLIB_FUNCTIONAL_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/core/syntax.h>

#define BCS_MAXIMUM_EWISE_ARGUMENTS 4

#define BCS_DECLARE_EWISE_FUNCTOR(Name, N) \
	template<typename T> struct is_ewise_functor<Name<T>, N> { static const bool value = true; }; \
	template<typename T> struct num_arguments<Name<T> > { static const bool value = N; };

#define BCS_DECLARE_REDUCTOR(Name, N) \
	template<typename T> struct is_reductor<Name<T>, N> { static const bool value = true; }; \
	template<typename T> struct num_arguments<Name<T> > { static const bool value = N; };


namespace bcs
{
	template<typename F> struct num_arguments;


	template<typename F, int N>
	struct is_ewise_functor
	{
		static const bool value = false;
	};

	template<typename F, int N>
	struct is_reductor
	{
		static const bool value = false;
	};

}

#endif /* FUNCTIONAL_H_ */
