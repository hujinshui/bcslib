/**
 * @file arg_check.h
 *
 * Facilities for argument checking
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARG_CHECK_H_
#define BCSLIB_ARG_CHECK_H_

#include <bcslib/base/basic_defs.h>

#include <type_traits>
#include <string>
#include <stdexcept>

namespace bcs
{

	// generic argument checking

	inline void check_arg(bool cond)
	{
		if (!cond)
		{
			throw std::invalid_argument("Invalid argument");
		}
	}


	inline void check_arg(bool cond, const std::string& message)
	{
		if (!cond)
		{
			throw std::invalid_argument(message);
		}
	}

	// index checking

	namespace _detail
	{
		template<typename TIndex, bool IsSigned> struct _index_check_helper;

		template<typename TIndex>
		struct _index_check_helper<TIndex, true>  // for signed index
		{
			static bool is_in_range(const TIndex& i, const TIndex& ub)
			{
				return i >= 0 && i < ub;
			}

			static bool is_in_range(const TIndex& i, const TIndex& lb, const TIndex& ub)
			{
				return i >= lb && i < ub;
			}
		};

		template<typename TIndex>
		struct _index_check_helper<TIndex, false>  // for unsigned index
		{
			static bool is_in_range(const TIndex& i, const TIndex& ub)
			{
				return i < ub;
			}

			static bool is_in_range(const TIndex& i, const TIndex& lb, const TIndex& ub)
			{
				return i >= lb && i < ub;
			}
		};
	}

	template<typename TIndex, typename TBound>
	inline bool is_index_in_range(const TIndex& i, const TBound& ub)
	{
		BCS_STATIC_ASSERT_V( std::is_integral<TIndex> );
		BCS_STATIC_ASSERT_V( std::is_integral<TBound> );

		return _detail::_index_check_helper<TIndex, std::is_signed<TIndex>::value>(
				i, static_cast<TIndex>(ub));
	}

	template<typename TIndex, typename TBound>
	inline bool is_index_in_range(const TIndex& i, const TBound& lb, const TBound& ub)
	{
		BCS_STATIC_ASSERT_V( std::is_integral<TIndex> );
		BCS_STATIC_ASSERT_V( std::is_integral<TBound> );

		return _detail::_index_check_helper<TIndex, std::is_signed<TIndex>::value>(
				i, static_cast<TIndex>(lb), static_cast<TIndex>(ub));
	}


}


#endif 
