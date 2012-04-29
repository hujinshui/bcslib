/**
 * @file matrix_subviews_helper.h
 *
 * The helper class for makeing subviews
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_SUBVIEWS_HELPER_H_
#define BCSLIB_MATRIX_SUBVIEWS_HELPER_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs { namespace detail {


	template<class Mat>
	struct slice_helper
	{
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;
		static const int comptime_rows = matrix_traits<Mat>::compile_time_num_rows;
		static const int comptime_cols = matrix_traits<Mat>::compile_time_num_cols;

		typedef typename matrix_traits<Mat>::value_type value_type;

		// view types

		typedef cref_matrix<value_type, comptime_rows, 1> const_column_type;
		typedef cref_matrix_ex<value_type, 1, comptime_cols> const_row_type;

		typedef typename select_type<is_readonly,
					cref_matrix<value_type, comptime_rows, 1>,
					ref_matrix<value_type, comptime_rows, 1>
				>::type column_type;

		typedef typename select_type<is_readonly,
					cref_matrix_ex<value_type, 1, comptime_cols>,
					ref_matrix_ex<value_type, 1, comptime_cols>
				>::type row_type;


		// view construction

		BCS_ENSURE_INLINE
		static const_column_type get_column(const IDenseMatrix<Mat, value_type>& mat, const index_t j)
		{
			return const_column_type(col_ptr(mat, j), mat.nrows(), 1);
		}

		BCS_ENSURE_INLINE
		static column_type get_column(IDenseMatrix<Mat, value_type>& mat, const index_t j)
		{
			return column_type(col_ptr(mat, j), mat.nrows(), 1);
		}

		BCS_ENSURE_INLINE
		static const_row_type get_row(const IDenseMatrix<Mat, value_type>& mat, const index_t i)
		{
			return const_row_type(row_ptr(mat, i), 1, mat.ncolumns(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static row_type get_row(IDenseMatrix<Mat, value_type>& mat, const index_t i)
		{
			return row_type(row_ptr(mat, i), 1, mat.ncolumns(), mat.lead_dim());
		}

	};


	template<class Mat, bool IsContinuous> struct multislice_helper;

	template<class Mat>
	struct multislice_helper<Mat, true>
	{
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;
		static const int comptime_rows = matrix_traits<Mat>::compile_time_num_rows;
		static const int comptime_cols = matrix_traits<Mat>::compile_time_num_cols;

		typedef typename matrix_traits<Mat>::value_type value_type;

		// view types

		typedef cref_matrix<value_type, comptime_rows, DynamicDim> const_multicolumn_type;
		typedef cref_matrix_ex<value_type, DynamicDim, comptime_cols> const_multirow_type;

		typedef typename select_type<is_readonly,
					cref_matrix<value_type, comptime_rows, DynamicDim>,
					ref_matrix<value_type, comptime_rows, DynamicDim>
				>::type multicolumn_type;

		typedef typename select_type<is_readonly,
					cref_matrix_ex<value_type, DynamicDim, comptime_cols>,
					ref_matrix_ex<value_type, DynamicDim, comptime_cols>
				>::type multirow_type;

		BCS_ENSURE_INLINE
		static const_multicolumn_type get_multicolumn(const IDenseMatrix<Mat, value_type>& mat,
				const index_t j, const index_t n)
		{
			return const_multicolumn_type(col_ptr(mat, j), mat.nrows(), n);
		}

		BCS_ENSURE_INLINE
		static multicolumn_type get_multicolumn(IDenseMatrix<Mat, value_type>& mat,
				const index_t j, const index_t n)
		{
			return multicolumn_type(col_ptr(mat, j), mat.nrows(), n);
		}

		BCS_ENSURE_INLINE
		static const_multirow_type get_multirow(const IDenseMatrix<Mat, value_type>& mat,
				const index_t i, const index_t m)
		{
			return const_multirow_type(row_ptr(mat, i), m, mat.ncolumns(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static multirow_type get_multirow(IDenseMatrix<Mat, value_type>& mat,
				const index_t i, const index_t m)
		{
			return multirow_type(row_ptr(mat, i), m, mat.ncolumns(), mat.lead_dim());
		}

	};


	template<class Mat>
	struct multislice_helper<Mat, false>
	{
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;
		static const int comptime_rows = matrix_traits<Mat>::compile_time_num_rows;
		static const int comptime_cols = matrix_traits<Mat>::compile_time_num_cols;

		typedef typename matrix_traits<Mat>::value_type value_type;

		// view types

		typedef cref_matrix_ex<value_type, comptime_rows, DynamicDim> const_multicolumn_type;
		typedef cref_matrix_ex<value_type, DynamicDim, comptime_cols> const_multirow_type;

		typedef typename select_type<is_readonly,
					cref_matrix_ex<value_type, comptime_rows, DynamicDim>,
					ref_matrix_ex<value_type, comptime_rows, DynamicDim>
				>::type multicolumn_type;

		typedef typename select_type<is_readonly,
					cref_matrix_ex<value_type, DynamicDim, comptime_cols>,
					ref_matrix_ex<value_type, DynamicDim, comptime_cols>
				>::type multirow_type;

		BCS_ENSURE_INLINE
		static const_multicolumn_type get_multicolumn(const IDenseMatrix<Mat, value_type>& mat,
				const index_t j, const index_t n)
		{
			return const_multicolumn_type(col_ptr(mat, j), mat.nrows(), n, mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static multicolumn_type get_multicolumn(IDenseMatrix<Mat, value_type>& mat,
				const index_t j, const index_t n)
		{
			return multicolumn_type(col_ptr(mat, j), mat.nrows(), n, mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static const_multirow_type get_multirow(const IDenseMatrix<Mat, value_type>& mat,
				const index_t i, const index_t m)
		{
			return const_multirow_type(row_ptr(mat, i), m, mat.ncolumns(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static multirow_type get_multirow(IDenseMatrix<Mat, value_type>& mat,
				const index_t i, const index_t m)
		{
			return multirow_type(row_ptr(mat, i), m, mat.ncolumns(), mat.lead_dim());
		}

	};




} }

#endif /* MATRIX_SUBVIEWS_HELPER_H_ */
