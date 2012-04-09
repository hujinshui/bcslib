/**
 * @file array2d.h
 *
 * Two dimensional array classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY2D_H
#define BCSLIB_ARRAY2D_H


#include <bcslib/array/aview2d.h>
#include <bcslib/array/aview2d_ops.h>
#include <bcslib/array/array1d.h>

#include <bcslib/array/transpose2d.h>
#include <bcslib/base/type_traits.h>

#include <cmath>

namespace bcs
{

	template<typename T, class Alloc=aligned_allocator<T> > class array2d;

	template<typename T, class Alloc>
	struct aview_traits<array2d<T, Alloc> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)
	};

	template<typename T, class Alloc>
	class array2d
	: public IConstContinuousAView2D<array2d<T, Alloc>, T>
	, public IContinuousAView2D<array2d<T, Alloc>, T>
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)

		typedef block<T, Alloc> block_type;

	public:
		explicit array2d(index_type m, index_type n)
		: m_block(m * n)
		, m_nrows(m), m_ncols(n)
		{
		}

		explicit array2d(const shape_type& shape)
		: m_block(shape[0] * shape[1])
		, m_nrows(shape[0]), m_ncols(shape[1])
		{
		}

		array2d(index_type m, index_type n, const T& x)
		: m_block(m * n, x)
		, m_nrows(m), m_ncols(n)
		{
		}

		array2d(index_type m, index_type n, const_pointer src)
		: m_block(m * n, src)
		, m_nrows(m), m_ncols(n)
		{
		}

		array2d(const array2d& r)
		: m_block(r.m_block)
		, m_nrows(r.nrows()), m_ncols(r.ncolumns())
		{
		}

		template<class Derived>
		explicit array2d(const IConstRegularAView2D<Derived, T>& r)
		: m_block(r.nelems())
		, m_nrows(r.nrows()), m_ncols(r.ncolumns())
		{
			copy(r.derived(), *this);
		}

		array2d& operator = (const array2d& r)
		{
			if (this != &r)
			{
				m_block = r.m_block;
				m_nrows = r.m_nrows;
				m_ncols = r.m_ncols;
			}
			return *this;
		}

		void swap(array2d& r)
		{
			using std::swap;

			m_block.swap(r.m_block);
			swap(m_nrows, r.m_nrows);
			swap(m_ncols, r.m_ncols);
		}

		caview2d<T> cview() const
		{
			return caview2d<T>(pbase(), nrows(), ncolumns());
		}

		aview2d<T> view()
		{
			return aview2d<T>(pbase(), nrows(), ncolumns());
		}

	public:
		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)m_block.nelems();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_block.nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nelems() == 0;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_nrows, m_ncols);
		}


	public:
		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_block.pbase();
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return m_block.pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return m_block[i];
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return m_block[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_block[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return m_block[offset(i, j)];
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return i + m_nrows * j;
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		row_view_type row(index_t i)
		{
			return row_view(*this, i);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		column_view_type column(index_t i)
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::view_type
		row(index_t i, const TRange& rgn)
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, true, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, true, TRange0, TRange1>::view_type
		V(const TRange0& I, const TRange1& J)
		{
			return subview(*this, I, J);
		}

		caview1d<T> flatten() const
		{
			return caview1d<T>(pbase(), nelems());
		}

		aview1d<T> flatten()
		{
			return aview1d<T>(pbase(), nelems());
		}

	private:
		block_type m_block;
		index_type m_nrows;
		index_type m_ncols;

	}; // end class array2d


	template<typename T, class Alloc>
	inline void swap(array2d<T, Alloc>& lhs, array2d<T, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	template<class Derived, typename T, class IndexSelector>
	array1d<T> select_elems(const IConstRegularAView2D<Derived, T>& a, const IndexSelector& I, const IndexSelector& J)
	{
		index_t n = (index_t)I.size();
		check_arg(n == (index_t)J.size(), "Inconsistent selector sizes.");

		array1d<T> r(n);

		const Derived& ad = a.derived();

		for (index_t i = 0; i < n; ++i)
		{
			r[i] = ad(I[i], J[i]);
		}
		return r;
	}


	template<class Derived, typename T, class IndexSelector>
	array2d<T> select_rows(const IConstRegularAView2D<Derived, T>& a, const IndexSelector& I)
	{
		index_t m = (index_t)I.size();
		index_t n = a.ncolumns();
		array2d<T> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m; ++i)
			{
				*(rp++) = ad(I[i], j);
			}
		}

		return r;
	}


	template<class Derived, typename T, class IndexSelector>
	array2d<T> select_columns(const IConstRegularAView2D<Derived, T>& a, const IndexSelector& J)
	{
		index_t m = a.nrows();
		index_t n = (index_t)J.size();
		array2d<T> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		for (index_t j = 0; j < n; ++j)
		{
			index_t sj = J[j];
			for (index_t i = 0; i < m; ++i)
			{
				*(rp++) = ad(i, sj);
			}
		}

		return r;
	}


	template<class Derived, typename T, class IndexSelectorI, class IndexSelectorJ>
	array2d<T> select_rows_cols(const IConstRegularAView2D<Derived, T>& a, const IndexSelectorI& I, const IndexSelectorJ& J)
	{
		index_t m = (index_t)I.size();
		index_t n = (index_t)J.size();
		array2d<T> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m; ++i)
			{
				*(rp++) = ad(I[i], J[j]);
			}
		}

		return r;
	}


	/******************************************************
	 *
	 *  Transposition
	 *
	 ******************************************************/

	template<typename T>
	inline void transpose_matrix(const T *src, T *dst, index_t m, index_t n)
	{
		const size_t block_size = BCS_TRANSPOSITION_BLOCK_BYTES / sizeof(T);

		T cache[block_size];

		if (block_size < 4)
		{
			direct_transpose_matrix(src, dst, (size_t)m, (size_t)n);
		}
		else
		{
			size_t bdim = (size_t)std::sqrt((double)block_size);
			blockwise_transpose_matrix(src, dst, (size_t)m, (size_t)n, bdim, cache);
		}
	}

	template<class Derived, typename T>
	inline array2d<T> transpose(const IConstContinuousAView2D<Derived, T>& a)
	{
		array2d<T> r(a.ncolumns(), a.nrows());

		transpose_matrix(a.pbase(), r.pbase(), a.ncolumns(), a.nrows());

		return r;
	}

}


#endif



