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
#include <bcslib/matrix/bits/ref_matrix_internal.h>

namespace bcs
{

	/********************************************
	 *
	 *  cref_matrix
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<cref_matrix<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = true;
		static const bool is_sparse = false;
		static const bool is_readonly = true;
		static const bool is_lazy = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class cref_matrix : public IDenseMatrix<cref_matrix<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_CDEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;

	public:
		cref_matrix(const T* pdata, index_type m, index_type n)
		: m_internal(pdata, m, n)
		{
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a cref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal<const T, CTRows, CTCols> m_internal;

	}; // end class cref_matrix




	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<ref_matrix<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = true;
		static const bool is_sparse = false;
		static const bool is_readonly = false;
		static const bool is_lazy = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class ref_matrix : public IDenseMatrix<ref_matrix<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;

	public:
		ref_matrix(T* pdata, index_type m, index_type n)
		: m_internal(pdata, m, n)
		{
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE pointer ptr_data()
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type i)
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a ref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal<T, CTRows, CTCols> m_internal;

	}; // end ref_matrix



}


#endif /* REF_MATRIX_H_ */






