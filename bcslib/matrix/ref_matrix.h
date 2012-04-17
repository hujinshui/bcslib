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

	public:

		BCS_ENSURE_INLINE
		explicit RefMatrix()
		: m_base(BCS_NULL), m_nrows(RowDim), m_ncols(ColDim)
		{
		}

		BCS_ENSURE_INLINE
		RefMatrix(T *data, index_type m, index_type n)
		{
			reset(data, m, n);
		}

		BCS_ENSURE_INLINE
		void reset(T *data, index_type m, index_type n)
		{
			if (RowDim >= 1) check_arg(m == RowDim);
			if (ColDim >= 1) check_arg(n == ColDim);

			m_base = data;
			m_nrows = m;
			m_ncols = n;
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		const RefMatrix& operator = (IMatrixBase<OtherDerived, T>& other)
		{
			detail::check_assign_fits(*this, other);
			assign(other);
			return *this;
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		void assign(IMatrixBase<OtherDerived, T>& other)
		{
			evaluate_to(other, *this);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base())
			{
				bcs::copy(*this, dst);
			}
		}

		BCS_ENSURE_INLINE void move_forward(index_t x)
		{
			m_base += x;
		}

		BCS_ENSURE_INLINE void move_backward(index_t x)
		{
			m_base -= x;
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

		BCS_ENSURE_INLINE bool is_vector() const
		{
			return nrows() == 1 || ncolumns() == 1;
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
	};



	/********************************************
	 *
	 *  CRefMatrix
	 *
	 ********************************************/

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

	public:

		BCS_ENSURE_INLINE
		explicit CRefMatrix()
		: m_base(BCS_NULL), m_nrows(RowDim), m_ncols(ColDim)
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const T *data, index_type m, index_type n)
		{
			reset(data, m, n);
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const CRefMatrix& other)
		: m_base(other.m_base), m_nrows(other.m_nrows), m_ncols(other.m_ncols)
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const RefMatrix<T, RowDim, ColDim>& other)
		: m_base(other.ptr_base()), m_nrows(other.nrows()), m_ncols(other.ncolumns())
		{
		}

		BCS_ENSURE_INLINE
		void reset(const T *data, index_type m, index_type n)
		{
			if (RowDim >= 1) check_arg(m == RowDim);
			if (ColDim >= 1) check_arg(n == ColDim);

			m_base = data;
			m_nrows = m;
			m_ncols = n;
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base())
			{
				bcs::copy(*this, dst);
			}
		}

		BCS_ENSURE_INLINE void move_forward(index_t x)
		{
			m_base += x;
		}

		BCS_ENSURE_INLINE void move_backward(index_t x)
		{
			m_base -= x;
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

		BCS_ENSURE_INLINE bool is_vector() const
		{
			return nrows() == 1 || ncolumns() == 1;
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


	/********************************************
	 *
	 *  Ref/CRef Vectors
	 *
	 ********************************************/

	template<typename T, int RowDim>
	class RefCol : public RefMatrix<T, RowDim, 1>
	{
		typedef RefMatrix<T, RowDim, 1> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = RowDim;
		static const index_t ColDimension = 1;

	public:

		BCS_ENSURE_INLINE
		explicit RefCol() : base_mat_t() { }

		BCS_ENSURE_INLINE
		RefCol(const base_mat_t& other) : base_mat_t(other) { }

		BCS_ENSURE_INLINE
		RefCol(T *data, index_t m) : base_mat_t(data, m, 1) { }

	}; // end class RefCol


	template<typename T, int RowDim>
	class CRefCol : public CRefMatrix<T, RowDim, 1>
	{
		typedef CRefMatrix<T, RowDim, 1> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = RowDim;
		static const index_t ColDimension = 1;

	public:

		BCS_ENSURE_INLINE
		explicit CRefCol() : base_mat_t() { }

		BCS_ENSURE_INLINE
		CRefCol(const base_mat_t& other) : base_mat_t(other) { }

		BCS_ENSURE_INLINE
		CRefCol(const T *data, index_t m) : base_mat_t(data, m, 1) { }

	}; // end class CRefCol



	template<typename T, int ColDim>
	class RefRow : public RefMatrix<T, 1, ColDim>
	{
		typedef RefMatrix<T, 1, ColDim> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = 1;
		static const index_t ColDimension = ColDim;

	public:

		BCS_ENSURE_INLINE
		explicit RefRow() : base_mat_t() { }

		BCS_ENSURE_INLINE
		RefRow(const base_mat_t& other) : base_mat_t(other) { }

		BCS_ENSURE_INLINE
		RefRow(T *data, index_t m) : base_mat_t(data, 1, m) { }

	}; // end class RefRow


	template<typename T, int ColDim>
	class CRefRow : public CRefMatrix<T, 1, ColDim>
	{
		typedef CRefMatrix<T, 1, ColDim> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = 1;
		static const index_t ColDimension = ColDim;

	public:

		BCS_ENSURE_INLINE
		explicit CRefRow() : base_mat_t() { }

		BCS_ENSURE_INLINE
		CRefRow(const base_mat_t& other) : base_mat_t(other) { }

		BCS_ENSURE_INLINE
		CRefRow(T *data, index_t m) : base_mat_t(data, 1, m) { }

	}; // end class CRefRow


}


#endif /* REF_MATRIX_H_ */






