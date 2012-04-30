/**
 * @file functor_base.h
 *
 * Some basic definitions for functors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_FUNCTOR_BASE_H_
#define BCSLIB_FUNCTOR_BASE_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/utils/arg_check.h>

namespace bcs
{
	template<typename Fun>
	struct is_unary_ewise_functor
	{
		static const bool value = false;
	};

	template<typename Fun>
	struct is_binary_ewise_functor
	{
		static const bool value = false;
	};

	template<typename Fun>
	struct is_unary_reduction_functor
	{
		static const bool value = false;
	};

	template<typename Fun>
	struct is_binary_reduction_functor
	{
		static const bool value = false;
	};

}


#define DECLARE_UNARY_EWISE_FUNCTOR(Name) \
	template<typename T> struct Name; \
	template<typename T> \
	struct is_unary_ewise_functor<Name<T> > { static const bool value = true; };

#define DECLARE_BINARY_EWISE_FUNCTOR(Name) \
	template<typename T> struct Name; \
	template<typename T> \
	struct is_binary_ewise_functor<Name<T> > { static const bool value = true; };

#define DECLARE_UNARY_REDUCTION_FUNCTOR(Name) \
	template<typename T> struct Name; \
	template<typename T> \
	struct is_unary_reduction_functor<Name<T> > { static const bool value = true; };

#define DECLARE_BINARY_REDUCTION_FUNCTOR(Name) \
	template<typename T> struct Name; \
	template<typename T> \
	struct is_binary_reduction_functor<Name<T> > { static const bool value = true; };

#endif





