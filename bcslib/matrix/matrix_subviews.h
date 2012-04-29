/**
 * @file matrix_subviews.h
 *
 * Functions for sub-views and slices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_SUBVIEWS_H_
#define BCSLIB_MATRIX_SUBVIEWS_H_

#include <bcslib/matrix/bits/matrix_subviews_helper.h>

namespace bcs
{

	template<class Mat>
	struct subviews
	{
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;
		static const int comptime_rows = matrix_traits<Mat>::compile_time_num_rows;
		static const int comptime_cols = matrix_traits<Mat>::compile_time_num_cols;

		typedef typename matrix_traits<Mat>::value_type value_type;


		// slices

		typedef typename detail::slice_helper<Mat>::const_row_type 		const_row_type;
		typedef typename detail::slice_helper<Mat>::row_type 			row_type;
		typedef typename detail::slice_helper<Mat>::const_column_type 	const_column_type;
		typedef typename detail::slice_helper<Mat>::column_type 		column_type;

		BCS_ENSURE_INLINE
		static const_row_type get_row(const Mat& mat, const index_t i)
		{
			return detail::slice_helper<Mat>::get_row(mat, i);
		}

		BCS_ENSURE_INLINE
		static row_type get_row(Mat& mat, const index_t i)
		{
			return detail::slice_helper<Mat>::get_row(mat, i);
		}

		BCS_ENSURE_INLINE
		static const_column_type get_column(const Mat& mat, const index_t j)
		{
			return detail::slice_helper<Mat>::get_column(mat, j);
		}

		BCS_ENSURE_INLINE
		static column_type get_column(Mat& mat, const index_t j)
		{
			return detail::slice_helper<Mat>::get_column(mat, j);
		}

	};


}

#endif
