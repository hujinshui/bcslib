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

#include <bcslib/matrix/matrix_ctf.h>

namespace bcs
{

	/********************************************
	 *
	 *  column views
	 *
	 ********************************************/

	template<class Mat>
	struct colviews<Mat, whole>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;

		static const int ctrows = ct_rows<Mat>::value;

		typedef cref_matrix<value_type, ctrows, 1> const_type;
		typedef  ref_matrix<value_type, ctrows, 1> non_const_type;

		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const index_t j, whole)
		{
			return const_type(col_ptr(mat, j), mat.nrows(), 1);
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const index_t j, whole)
		{
			return type(col_ptr(mat, j), mat.nrows(), 1);
		}
	};


	template<class Mat>
	struct colviews<Mat, range>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;

		typedef cref_matrix<value_type, DynamicDim, 1> const_type;
		typedef  ref_matrix<value_type, DynamicDim, 1> non_const_type;

		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const index_t j, const range &rg)
		{
			return const_type(ptr_elem(mat, rg.begin_index(), j), rg.num(), 1);
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const index_t j, const range& rg)
		{
			return type(ptr_elem(mat, rg.begin_index(), j), rg.num(), 1);
		}
	};



	/********************************************
	 *
	 *  row views
	 *
	 ********************************************/

	template<class Mat>
	struct rowviews<Mat, whole>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;

		static const int ctcols = ct_cols<Mat>::value;

		typedef cref_matrix_ex<value_type, 1, ctcols> const_type;
		typedef  ref_matrix_ex<value_type, 1, ctcols> non_const_type;

		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const index_t i, whole)
		{
			return const_type(row_ptr(mat, i), 1, mat.ncolumns(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const index_t i, whole)
		{
			return type(row_ptr(mat, i), 1, mat.ncolumns(), mat.lead_dim());
		}

	};


	template<class Mat>
	struct rowviews<Mat, range>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;

		typedef cref_matrix_ex<value_type, 1, DynamicDim> const_type;
		typedef  ref_matrix_ex<value_type, 1, DynamicDim> non_const_type;

		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const index_t i, const range& rg)
		{
			return const_type(ptr_elem(mat, i, rg.begin_index()), 1, rg.num(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const index_t i, const range& rg)
		{
			return type(ptr_elem(mat, i, rg.begin_index()), 1, rg.num(), mat.lead_dim());
		}

	};



	/********************************************
	 *
	 *  subviews
	 *
	 ********************************************/

	namespace detail
	{
		template<class Mat, int CTCols, bool IsCont> struct multicol_helper;

		template<class Mat, int CTCols>
		struct multicol_helper<Mat, CTCols, true>
		{
			typedef typename matrix_traits<Mat>::value_type value_type;
			static const bool is_readonly = matrix_traits<Mat>::is_readonly;
			static const int ctrows = ct_rows<Mat>::value;

			typedef cref_matrix<value_type, ctrows, CTCols> const_type;
			typedef  ref_matrix<value_type, ctrows, CTCols> non_const_type;
			typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

			BCS_ENSURE_INLINE
			static const_type get(const Mat& mat, const index_t j, const index_t n)
			{
				return const_type(col_ptr(mat, j), mat.nrows(), n);
			}

			BCS_ENSURE_INLINE
			static type get(Mat& mat, const index_t j, const index_t n)
			{
				return type(col_ptr(mat, j), mat.nrows(), n);
			}
		};


		template<class Mat, int CTCols>
		struct multicol_helper<Mat, CTCols, false>
		{
			typedef typename matrix_traits<Mat>::value_type value_type;
			static const bool is_readonly = matrix_traits<Mat>::is_readonly;
			static const int ctrows = ct_rows<Mat>::value;

			typedef cref_matrix_ex<value_type, ctrows, CTCols> const_type;
			typedef  ref_matrix_ex<value_type, ctrows, CTCols> non_const_type;
			typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

			BCS_ENSURE_INLINE
			static const_type get(const Mat& mat, const index_t j, const index_t n)
			{
				return const_type(col_ptr(mat, j), mat.nrows(), n, mat.lead_dim());
			}

			BCS_ENSURE_INLINE
			static type get(Mat& mat, const index_t j, const index_t n)
			{
				return type(col_ptr(mat, j), mat.nrows(), n, mat.lead_dim());
			}
		};
	}

	template<class Mat>
	struct subviews<Mat, whole, whole>
	{
		static const bool is_continuous = has_continuous_layout<Mat>::value;
		typedef detail::multicol_helper<Mat, ct_cols<Mat>::value, is_continuous> helper_t;

		typedef typename helper_t::const_type const_type;
		typedef typename helper_t::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, whole, whole)
		{
			return helper_t::get(mat, 0, mat.ncolumns());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, whole, whole)
		{
			return helper_t::get(mat, 0, mat.ncolumns());
		}
	};


	template<class Mat>
	struct subviews<Mat, whole, range>
	{
		static const bool is_continuous = has_continuous_layout<Mat>::value;
		typedef detail::multicol_helper<Mat, DynamicDim, is_continuous> helper_t;

		typedef typename helper_t::const_type const_type;
		typedef typename helper_t::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, whole, const range& rg)
		{
			return helper_t::get(mat, rg.begin_index(), rg.num());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, whole, const range& rg)
		{
			return helper_t::get(mat, rg.begin_index(), rg.num());
		}
	};


	template<class Mat>
	struct subviews<Mat, range, whole>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;
		static const int ctcols = ct_cols<Mat>::value;

		typedef cref_matrix_ex<value_type, DynamicDim, ctcols> const_type;
		typedef  ref_matrix_ex<value_type, DynamicDim, ctcols> non_const_type;
		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const range& rg, whole)
		{
			return const_type(row_ptr(mat, rg.begin_index()), rg.num(),
					mat.ncolumns(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const range& rg, whole)
		{
			return type(row_ptr(mat, rg.begin_index()), rg.num(),
					mat.ncolumns(), mat.lead_dim());
		}
	};


	template<class Mat>
	struct subviews<Mat, range, range>
	{
		typedef typename matrix_traits<Mat>::value_type value_type;
		static const bool is_readonly = matrix_traits<Mat>::is_readonly;

		typedef cref_matrix_ex<value_type, DynamicDim, DynamicDim> const_type;
		typedef  ref_matrix_ex<value_type, DynamicDim, DynamicDim> non_const_type;
		typedef typename select_type<is_readonly, const_type, non_const_type>::type type;

		BCS_ENSURE_INLINE
		static const_type get(const Mat& mat, const range& rrg, const range& crg)
		{
			return const_type(
					ptr_elem(mat, rrg.begin_index(), crg.begin_index()),
					rrg.num(), crg.num(), mat.lead_dim());
		}

		BCS_ENSURE_INLINE
		static type get(Mat& mat, const range& rrg, const range& crg)
		{
			return type(
					ptr_elem(mat, rrg.begin_index(), crg.begin_index()),
					rrg.num(), crg.num(), mat.lead_dim());
		}
	};

}

#endif
