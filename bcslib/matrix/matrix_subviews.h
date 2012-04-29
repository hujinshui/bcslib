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
		static const bool is_continuous = matrix_traits<Mat>::is_continuous;
		static const int comptime_rows = matrix_traits<Mat>::compile_time_num_rows;
		static const int comptime_cols = matrix_traits<Mat>::compile_time_num_cols;

		typedef typename matrix_traits<Mat>::value_type value_type;


		// slices

		typedef detail::slice_helper<Mat> slice_ht;

		typedef typename slice_ht::const_row_type 		const_row_type;
		typedef typename slice_ht::row_type 			row_type;
		typedef typename slice_ht::const_column_type	const_column_type;
		typedef typename slice_ht::column_type 			column_type;

		BCS_ENSURE_INLINE
		static const_row_type get_row(const Mat& mat, const index_t i)
		{
			return slice_ht::get_row(mat, i);
		}

		BCS_ENSURE_INLINE
		static row_type get_row(Mat& mat, const index_t i)
		{
			return slice_ht::get_row(mat, i);
		}

		BCS_ENSURE_INLINE
		static const_column_type get_column(const Mat& mat, const index_t j)
		{
			return slice_ht::get_column(mat, j);
		}

		BCS_ENSURE_INLINE
		static column_type get_column(Mat& mat, const index_t j)
		{
			return slice_ht::get_column(mat, j);
		}


		// multi-slices

		typedef detail::multislice_helper<Mat, is_continuous> multislice_ht;

		typedef typename multislice_ht::const_multirow_type 	const_multirow_type;
		typedef typename multislice_ht::multirow_type 			multirow_type;
		typedef typename multislice_ht::const_multicolumn_type 	const_multicolumn_type;
		typedef typename multislice_ht::multicolumn_type 		multicolumn_type;

		BCS_ENSURE_INLINE
		static const_multirow_type get_multirow(const Mat& mat, const index_t i, const index_t m)
		{
			return multislice_ht::get_multirow(mat, i, m);
		}

		BCS_ENSURE_INLINE
		static multirow_type get_multirow(Mat& mat, const index_t i, const index_t m)
		{
			return multislice_ht::get_multirow(mat, i, m);
		}

		BCS_ENSURE_INLINE
		static const_multicolumn_type get_multicolumn(const Mat& mat, const index_t j, const index_t n)
		{
			return multislice_ht::get_multicolumn(mat, j, n);
		}

		BCS_ENSURE_INLINE
		static multicolumn_type get_multicolumn(Mat& mat, const index_t j, const index_t n)
		{
			return multislice_ht::get_multicolumn(mat, j, n);
		}


	};


}

#endif
