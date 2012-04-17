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

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class RefMatrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class CRefMatrix;

	template<typename T, int RowDim=DynamicDim> class RefCol;
	template<typename T, int RowDim=DynamicDim> class CRefCol;

	template<typename T, int ColDim=DynamicDim> class RefRow;
	template<typename T, int ColDim=DynamicDim> class CRefRow;

	template<typename T, typename Dir=VertDir, int Dim=DynamicDim> class StepVector;
	template<typename T, typename Dir=VertDir, int Dim=DynamicDim> class CStepVector;

	template<typename T> class RefBlock;
	template<typename T> class CRefBlock;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class DenseMatrix;
	template<typename T, int RowDim=DynamicDim> class DenseCol;
	template<typename T, int ColDim=DynamicDim> class DenseRow;


	// manipulation functions


	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrixView<Derived1, T>& A, const IDenseMatrixView<Derived2, T>& B);

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrixBlock<Derived1, T>& A, const IDenseMatrixBlock<Derived2, T>& B);

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrix<Derived1, T>& A, const IDenseMatrix<Derived2, T>& B);

	template<typename T, class Derived>
	inline void fill(IDenseMatrixBlock<Derived, T>& X, const T& v);

	template<typename T, class Derived>
	inline void zero(IDenseMatrix<Derived, T>& X);

	template<typename T, class Derived>
	inline void zero(IDenseMatrixBlock<Derived, T>& X);

	template<typename T, class Derived>
	inline void copy_from_mem(IDenseMatrix<Derived, T>& dst, const T *src);

	template<typename T, class Derived>
	inline void copy_from_mem(IDenseMatrixBlock<Derived, T>& dst, const T *src);

	template<typename T, class LDerived, class RDerived>
	inline void copy(const IDenseMatrixBlock<LDerived, T>& src, IDenseMatrixBlock<RDerived, T>& dst);

	template<typename T, class LDerived, class RDerived>
	inline void copy(const IDenseMatrix<LDerived, T>& src, IDenseMatrix<RDerived, T>& dst);

	template<typename T, class Derived>
	void printf_mat(const char *fmt, const IDenseMatrixView<Derived, T>& X,
			const char *pre_line=BCS_NULL, const char *delim="\n");


}

#endif 
