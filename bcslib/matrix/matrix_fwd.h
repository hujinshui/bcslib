/*
 * @file matrix_fwd.h
 *
 * Forward declaration of important classes
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_FWD_H_
#define BCSLIB_MATRIX_FWD_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{

	// forward declaration of some important types

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_matrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class  ref_matrix;

	template<typename T, int CTRows=DynamicDim> class cref_col;
	template<typename T, int CTRows=DynamicDim> class  ref_col;
	template<typename T, int CTCols=DynamicDim> class cref_row;
	template<typename T, int CTCols=DynamicDim> class  ref_row;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_matrix_ex;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class  ref_matrix_ex;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_grid2d;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class  ref_grid2d;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class dense_matrix;
	template<typename T, int CTRows=DynamicDim> class dense_col;
	template<typename T, int CTCols=DynamicDim> class dense_row;


	// manipulation functions

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B);

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IDenseMatrix<LMat, T>& A, const IDenseMatrix<RMat, T>& B);

	template<typename T, class SMat, class DMat>
	inline void copy(const IDenseMatrix<SMat, T>& src, IDenseMatrix<DMat, T>& dst);

	template<typename T, class SMat, class DMat>
	inline void copy(const IMatrixView<SMat, T>& src, IRegularMatrix<DMat, T>& dst);

	template<typename T, class Mat>
	inline void fill(IDenseMatrix<Mat, T>& A, const T& v);

	template<typename T, class Mat>
	inline void zero(IDenseMatrix<Mat, T>& A);

	template<typename T, class Mat>
	void printf_mat(const char *fmt, const IMatrixView<Mat, T>& X,
			const char *pre_line=BCS_NULL, const char *delim="\n");

}

#endif 
