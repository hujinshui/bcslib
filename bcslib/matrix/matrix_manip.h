/*
 * @file matrix_manip.h
 *
 * Generic matrix manipulation functions
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_MANIP_H_
#define BCSLIB_MATRIX_MANIP_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/base/mem_op.h>
#include <cstdio>

namespace bcs
{

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrixView<Derived1, T>& A, const IDenseMatrixView<Derived2, T>& B)
	{
		if (is_same_size(A, B))
		{
			index_t m = A.nrows();
			index_t n = A.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					if (A.elem(i, j) != B.elem(i, j)) return false;
				}
			}
			return true;
		}
		else
		{
			return false;
		}
	}


	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrixBlock<Derived1, T>& A, const IDenseMatrixBlock<Derived2, T>& B)
	{
		if (is_same_size(A, B))
		{
			return elems_equal_2d((size_t)A.nrows(), (size_t)A.ncolumns(),
					A.ptr_base(), (size_t)A.lead_dim(), B.ptr_base(), (size_t)B.lead_dim());
		}
		else return false;
	}

	template<typename T, class Derived1, class Derived2>
	inline bool is_equal(const IDenseMatrix<Derived1, T>& A, const IDenseMatrix<Derived2, T>& B)
	{
		if (is_same_size(A, B))
		{
			return elems_equal(A.size(), A.ptr_base(), B.ptr_base());
		}
		else return false;
	}

	template<typename T, class Derived>
	inline void fill(IDenseMatrixBlock<Derived, T>& X, const T& v)
	{
		fill_elems_2d((size_t)X.nrows(), (size_t)X.ncolumns(), X.ptr_base(), (size_t)X.lead_dim(), v);
	}

	template<typename T, class Derived>
	inline void zero(IDenseMatrix<Derived, T>& X)
	{
		zero_elems(X.size(), X.ptr_base());
	}

	template<typename T, class Derived>
	inline void zero(IDenseMatrixBlock<Derived, T>& X)
	{
		zero_elems_2d((size_t)X.nrows(), (size_t)X.ncolumns(), X.ptr_base(), (size_t)X.lead_dim());
	}

	template<typename T, class Derived>
	inline void copy_from_mem(IDenseMatrix<Derived, T>& dst, const T *src)
	{
		copy_elems((size_t)dst.nelems(), src, dst.ptr_base());
	}

	template<typename T, class Derived>
	inline void copy_from_mem(IDenseMatrixBlock<Derived, T>& dst, const T *src)
	{
		copy_elems_2d((size_t)dst.nrows(), (size_t)dst.ncolumns(),
				src, (size_t)dst.nrows(), dst.ptr_base(), (size_t)dst.lead_dim());
	}

	template<typename T, class LDerived, class RDerived>
	inline void copy(const IDenseMatrixBlock<LDerived, T>& src, IDenseMatrixBlock<RDerived, T>& dst)
	{
		check_arg( is_same_size(src, dst) );
		copy_elems_2d((size_t)src.nrows(), (size_t)src.ncolumns(),
				src.ptr_base(), (size_t)src.lead_dim(),
				dst.ptr_base(), (size_t)dst.lead_dim());
	}

	template<typename T, class LDerived, class RDerived>
	inline void copy(const IDenseMatrix<LDerived, T>& src, IDenseMatrix<RDerived, T>& dst)
	{
		check_arg( is_same_size(src, dst) );
		copy_elems(src.size(), src.ptr_base(), dst.ptr_base());
	}

	template<typename T, class Derived>
	void printf_mat(const char *fmt, const IDenseMatrixView<Derived, T>& X, const char *pre_line, const char *delim)
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
			std::printf("%s", delim);
		}
	}


}


#endif /* MATRIX_MANIP_H_ */
