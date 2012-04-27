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

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class ref_matrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_matrix;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class ref_block2d;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_block2d;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class ref_grid2d;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_grid2d;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class dense_matrix;
	template<typename T, int RowDim=DynamicDim> class dense_col;
	template<typename T, int ColDim=DynamicDim> class dense_row;


	// manipulation functions

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IMatrixView<Derived1, T>& A, const IMatrixView<Derived2, T>& B);

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrix<Derived1, T>& A, const IDenseMatrix<Derived2, T>& B);

	template<typename T, class Derived>
	inline void fill(IDenseMatrix<Derived, T>& X, const T& v);

	template<typename T, class Derived>
	inline void zero(IDenseMatrix<Derived, T>& X);

	template<typename T, class Derived>
	inline void copy_from_mem(IDenseMatrix<Derived, T>& dst, const T *src);

	template<typename T, class LDerived, class RDerived>
	inline void copy(const IDenseMatrix<LDerived, T>& src, IDenseMatrix<RDerived, T>& dst);

	template<typename T, class Derived>
	void printf_mat(const char *fmt, const IMatrixView<Derived, T>& X,
			const char *pre_line=BCS_NULL, const char *delim="\n");


}

#endif 
