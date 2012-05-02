/**
 * @file offset_calculator.h
 *
 * Implementation of efficient offset calculation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_OFFSET_HELPER_H_
#define BCSLIB_OFFSET_HELPER_H_

#include <bcslib/matrix/matrix_fwd.h>

namespace bcs { namespace detail {

	template<bool IsRow, bool IsCol>
	struct offset_helper;

	template<> struct offset_helper<false, false>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t get(const Mat& a, const index_t i, const index_t j)
		{
			return i + a.lead_dim() * j;
		}
	};

	template<> struct offset_helper<false, true>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t get(const Mat& a, const index_t i, const index_t j)
		{
			return i;
		}
	};

	template<> struct offset_helper<true, false>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t get(const Mat& a, const index_t i, const index_t j)
		{
			return a.lead_dim() * j;
		}
	};

	template<> struct offset_helper<true, true>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t get(const Mat& a, const index_t i, const index_t j)
		{
			return 0;
		}
	};


	template<class Mat>
	BCS_ENSURE_INLINE
	index_t calc_offset(const Mat& a, const index_t i, const index_t j)
	{
		return offset_helper<ct_is_row<Mat>::value, ct_is_col<Mat>::value>::get(a, i, j);
	}


} }



#endif /* OFFSET_CALCULATOR_H_ */
