/**
 * @file matrix_properties.h
 *
 * Functions to get properties of Matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PROPERTIES_H_
#define BCSLIB_MATRIX_PROPERTIES_H_

#include <bcslib/matrix/matrix_ctf.h>
#include <bcslib/matrix/matrix_concepts.h>

namespace bcs
{
	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_empty(const IMatrixXpr<Mat, T>& X)
	{
		return (has_dynamic_nrows<Mat>::value && X.nrows() == 0) ||
			(has_dynamic_ncols<Mat>::value && X.ncolumns() == 0);
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_column(const IMatrixXpr<Mat, T>& X)
	{
		return ct_is_col<Mat>::value || X.ncolumns() == 1;
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_row(const IMatrixXpr<Mat, T>& X)
	{
		return ct_is_row<Mat>::value || X.nrows() == 1;
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_scalar(const IMatrixXpr<Mat, T>& X)
	{
		return is_column(X) && is_row(X);
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_vector(const IMatrixXpr<Mat, T>& X)
	{
		return is_column(X) || is_row(X);
	}

	template<class Mat1, typename T1, class Mat2, typename T2>
	BCS_ENSURE_INLINE
	inline bool has_same_nrows(const IMatrixXpr<Mat1, T1>& A, const IMatrixXpr<Mat2, T2>& B)
	{
		return ct_has_same_nrows<Mat1, Mat2>::value || A.nrows() == B.nrows();
	}

	template<class Mat1, typename T1, class Mat2, typename T2>
	BCS_ENSURE_INLINE
	inline bool has_same_ncolumns(const IMatrixXpr<Mat1, T1>& A, const IMatrixXpr<Mat2, T2>& B)
	{
		return ct_has_same_ncols<Mat1, Mat2>::value || A.ncolumns() == B.ncolumns();
	}

	template<class Mat1, typename T1, class Mat2, typename T2>
	BCS_ENSURE_INLINE
	inline bool has_same_size(const IMatrixXpr<Mat1, T1>& A, const IMatrixXpr<Mat2, T2>& B)
	{
		return has_same_nrows(A, B) && has_same_ncolumns(A, B);
	}

	template<class Mat1, typename T1, class Mat2, typename T2>
	BCS_ENSURE_INLINE
	inline void check_same_size(const IMatrixXpr<Mat1, T1>& A, const IMatrixXpr<Mat2, T2>& B, const char *msg)
	{
		check_arg(has_same_size(A, B), msg);
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Mat>::const_pointer
	col_ptr(const IDenseMatrix<Mat, T>& X, index_t j)
	{
		return X.ptr_data() + X.lead_dim() * j;
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Mat>::pointer
	col_ptr(IDenseMatrix<Mat, T>& X, index_t j)
	{
		return X.ptr_data() + X.lead_dim() * j;
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Mat>::const_pointer
	row_ptr(const IDenseMatrix<Mat, T>& X, index_t i)
	{
		return X.ptr_data() + i;
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Mat>::pointer
	row_ptr(IDenseMatrix<Mat, T>& X, index_t i)
	{
		return X.ptr_data() + i;
	}


	namespace detail
	{
		template<class Mat, bool IsLinearAccessible> struct linear_acc_helper;

		template<class Mat> struct linear_acc_helper<Mat, true>
		{
			BCS_ENSURE_INLINE
			typename matrix_traits<Mat>::value_type get_value(const Mat& mat, index_t idx)
			{
				return mat[idx];
			}
		};

		template<class Mat> struct linear_acc_helper<Mat, false>
		{
			BCS_ENSURE_INLINE
			typename matrix_traits<Mat>::value_type get_value(const Mat& mat, index_t idx)
			{
				throw invalid_operation(
						"Attempted to access an non-linear-accessible matrix in a linear fashion");
			}
		};
	}


	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T get_by_linear_index(const IMatrixView<Mat, T>& mat, index_t idx)
	{
		return detail::linear_acc_helper<Mat,
				is_linear_accessible<Mat>::value>::get_value(mat.derived(), idx);
	}


}

#endif /* MATRIX_PROPERTIES_H_ */
