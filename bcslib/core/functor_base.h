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

	// convenient functions

	namespace detail
	{
		template<class Reductor, bool AllowEmpty> struct empty_reduc_result_of;

		template<class Reductor>
		struct empty_reduc_result_of<Reductor, true>
		{
			BCS_ENSURE_INLINE
			static typename Reductor::result_type
			get(const Reductor& reduc) { return reduc(); }
		};

		template<class Reductor>
		struct empty_reduc_result_of<Reductor, false>
		{
			BCS_ENSURE_INLINE
			static typename Reductor::result_type
			get(const Reductor& reduc)
			{ throw bcs::invalid_argument("Attempted to reduce on empty matrix."); }
		};
	}

	template<class Reductor>
	struct empty_reduc_result_of
	{
		BCS_ENSURE_INLINE
		static typename Reductor::result_type
		get(const Reductor& reduc)
		{
			return detail::empty_reduc_result_of<
					Reductor, Reductor::has_empty_value>::get(reduc);
		}
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





