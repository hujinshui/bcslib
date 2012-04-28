/**
 * @file matrix_manip.h
 *
 * Manipulation functions for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_MANIP_H_
#define BCSLIB_MATRIX_MANIP_H_

#include <bcslib/matrix/matrix_fwd.h>
#include <bcslib/matrix/bits/matrix_manip_helpers.h>
#include <cstdio>

namespace bcs
{

	// is_equal

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B)
	{
		if (is_same_size(A, B))
		{
			index_t m = A.nrows();
			index_t n = A.ncolumns();

			for (index_t j = 0; j < n; ++j)
				for (index_t i = 0; i < m; ++i)
					if (A.elem(i, j) != B.elem(i, j)) return false;

			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	inline bool is_equal(const IDenseMatrix<LMat, T>& A, const IDenseMatrix<RMat, T>& B)
	{
		if (is_same_size(A, B))
		{
			typedef detail::matrix_equal_helper<T,
					binary_ct_rows<LMat, RMat>::value,
					binary_ct_cols<LMat, RMat>::value> helper_t;

			return helper_t::run(A.nrows(), A.ncolumns(),
					A.ptr_data(), A.lead_dim(), B.ptr_data(), B.lead_dim());
		}
		else
		{
			return false;
		}
	}


	// copy

	template<typename T, class SMat, class DMat>
	inline void copy(const IMatrixView<SMat, T>& src, IRegularMatrix<DMat, T>& dst)
	{
		check_arg(is_same_size(src, dst), "The sizes of source and target are inconsistent.");

		index_t m = src.nrows();
		index_t n = src.ncolumns();

		for (index_t j = 0; j < n; ++j)
			for (index_t i = 0; i < m; ++i)
				dst.elem(i, j) = src.elem(i, j);
	}

	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void copy(const IDenseMatrix<SMat, T>& src, IDenseMatrix<DMat, T>& dst)
	{
		check_arg(is_same_size(src, dst), "The sizes of source and target are inconsistent.");

		typedef detail::matrix_copy_helper<T,
				binary_ct_rows<SMat, DMat>::value,
				binary_ct_cols<SMat, DMat>::value> helper_t;

		helper_t::run(src.nrows(), src.ncolumns(),
				src.ptr_data(), src.lead_dim(), dst.ptr_data(), dst.lead_dim());
	}


	// fill

	template<typename T, class Mat>
	inline void fill(IRegularMatrix<Mat, T>& A, const T& v)
	{
		index_t m = A.nrows();
		index_t n = A.ncolumns();

		for (index_t j = 0; j < n; ++j)
			for (index_t i = 0; i < m; ++i)
				A.elem(i, j) = v;
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	inline void fill(IDenseMatrix<Mat, T>& A, const T& v)
	{
		typedef detail::matrix_fill_helper<T,
				ct_rows<Mat>::value, ct_cols<Mat>::value> helper_t;

		helper_t::run(A.nrows(), A.ncolumns(), A.ptr_data(), A.lead_dim(), v);
	}


	// zero

	template<typename T, class Mat>
	inline void zero(IRegularMatrix<Mat, T>& A)
	{
		index_t m = A.nrows();
		index_t n = A.ncolumns();

		for (index_t j = 0; j < n; ++j)
			for (index_t i = 0; i < m; ++i)
				A.elem(i, j) = T(0);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	inline void zero(IDenseMatrix<Mat, T>& A)
	{
		typedef detail::matrix_zero_helper<T,
				ct_rows<Mat>::value, ct_cols<Mat>::value> helper_t;

		helper_t::run(A.nrows(), A.ncolumns(), A.ptr_data(), A.lead_dim());
	}



	// printf_mat

	template<typename T, class Mat>
	void printf_mat(const char *fmt, const IMatrixView<Mat, T>& X, const char *pre_line, const char *delim)
	{
		index_t m = X.nrows();
		index_t n = X.ncolumns();

		for (index_t i = 0; i < m; ++i)
		{
			if (pre_line) std::printf("%s", pre_line);
			for (index_t j = 0; j < n; ++j)
			{
				std::printf(fmt, X.elem(i, j));
			}
			if (delim) std::printf("%s", delim);
		}
	}




}

#endif /* MATRIX_MANIP_H_ */
