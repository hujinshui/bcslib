/**
 * @file matrix_assign.h
 *
 * The facility to support matrix assignment
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_ASSIGN_H_
#define BCSLIB_MATRIX_ASSIGN_H_

#include <bcslib/matrix/matrix_fwd.h>
#include <bcslib/matrix/matrix_xpr.h>

namespace bcs
{
	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixView<SMat, T>& src, IRegularMatrix<DMat, T>& dst)
	{
		copy(src.derived(), dst.derived());
	}

	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixView<SMat, T>& src, IDenseMatrix<DMat, T>& dst)
	{
		dst.resize(src.nrows(), src.ncolumns());
		copy(src.derived(), dst.derived());
	}


}

#endif /* MATRIX_ASSIGN_H_ */
