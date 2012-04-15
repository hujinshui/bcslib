/**
 * @file ref_matrix.h
 *
 * The reference matrix/vector/block classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_MATRIX_H_
#define BCSLIB_REF_MATRIX_H_

#include <bcslib/matrix/matrix_fwd.h>
#include "bits/matrix_helpers.h"

namespace bcs
{



	/********************************************
	 *
	 *  RefMatrix
	 *
	 ********************************************/

	template<typename T, int RowDim, int ColDim>
	struct matrix_traits<RefMatrix<T, RowDim, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		static const bool IsReadOnly = false;

		typedef const RefMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class RefMatrix : public IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T>
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const RefMatrix<T, RowDim, ColDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit RefMatrix()
		: m_base(BCS_NULL), m_nrows(RowDim), m_ncols(ColDim)
		{
		}

		BCS_ENSURE_INLINE
		RefMatrix(T *data, index_type m, index_type n)
		: m_base(data), m_nrows(m), m_ncols(n)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		const RefMatrix& operator = (IMatrixBase<OtherDerived, T>& other)
		{
			check_arg( is_same_size(*this, other) );
			other.eval_to(*this);
			return *this;
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base())
			{
				bcs::copy(*this, dst);
			}
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			bcs::copy(*this, dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nrows() == 0 || ncolumns() == 0;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_base;
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return m_base;
		}

		BCS_ENSURE_INLINE const_pointer col_ptr(index_type j) const
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE pointer col_ptr(index_type j)
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset<RowDim, ColDim>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_base[idx];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return m_base[idx];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	public:

		BCS_ENSURE_INLINE void zero()
		{
			bcs::zero(*this);
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			bcs::fill(*this, v);
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			bcs::copy_from_mem(*this, src);
		}

	private:
		T *m_base;
		index_t m_nrows;
		index_t m_ncols;

	}; // end class RefMatrix



	template<typename T, int RowDim, int ColDim>
	struct matrix_traits<CRefMatrix<T, RowDim, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		static const bool IsReadOnly = true;

		typedef const CRefMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class CRefMatrix : public IDenseMatrix<CRefMatrix<T, RowDim, ColDim>, T>
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<CRefMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const CRefMatrix<T, RowDim, ColDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit CRefMatrix()
		: m_base(BCS_NULL), m_nrows(RowDim), m_ncols(ColDim)
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const T *data, index_type m, index_type n)
		: m_base(data), m_nrows(m), m_ncols(n)
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(CRefMatrix& other)
		: m_base(other.m_base), m_nrows(other.m_nrows), m_ncols(other.m_ncols)
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const RefMatrix<T, RowDim, ColDim>& other)
		: m_base(other.ptr_base()), m_nrows(other.nrows()), m_ncols(other.ncolumns())
		{
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base())
			{
				bcs::copy(*this, dst);
			}
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			bcs::copy(*this, dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

	private:
		// no assignment
		CRefMatrix& operator = (const CRefMatrix& );


	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nrows() == 0 || ncolumns() == 0;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_base;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base()
		{
			return m_base;
		}

		BCS_ENSURE_INLINE const_pointer col_ptr(index_type j) const
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE const_pointer col_ptr(index_type j)
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset<RowDim, ColDim>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j)
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_base[idx];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx)
		{
			return m_base[idx];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j)
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	public:

		BCS_ENSURE_INLINE void zero() BCS_CREF_NOWRITE

		BCS_ENSURE_INLINE void fill(const_reference v) BCS_CREF_NOWRITE

		BCS_ENSURE_INLINE void copy_from(const_pointer src) BCS_CREF_NOWRITE

	private:
		const T *m_base;
		index_t m_nrows;
		index_t m_ncols;

	}; // end class CRefMatrix


}


#endif /* REF_MATRIX_H_ */





