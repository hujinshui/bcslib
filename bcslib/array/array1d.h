/**
 * @file array1d.h
 *
 * one-dimensional array
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY1D_H
#define BCSLIB_ARRAY1D_H

#include <bcslib/array/aview1d.h>
#include <bcslib/array/aview1d_ops.h>
#include <bcslib/base/block.h>

namespace bcs
{

	template<typename T, class Alloc=aligned_allocator<T> > class array1d;

	template<typename T, class Alloc>
	struct aview_traits<array1d<T, Alloc> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T)
	};

	template<typename T, class Alloc>
	class array1d
	: public IConstContinuousAView1D<array1d<T, Alloc>, T>
	, public IContinuousAView1D<array1d<T, Alloc>, T>
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(1u, T)

		typedef block<T, Alloc> block_type;

	public:
		explicit array1d(index_type n)
		: m_block(n)
		{
		}

		explicit array1d(const shape_type& shape)
		: m_block(shape[0])
		{
		}

		array1d(index_type n, const value_type& v)
		: m_block(n, v)
		{
		}

		array1d(index_type n, const_pointer src)
		: m_block(n, src)
		{
		}

		array1d(const array1d& r)
		: m_block(r.m_block)
		{
		}

		template<class Derived>
		explicit array1d(const IConstRegularAView1D<Derived, T>& r)
		: m_block(r.nelems())
		{
			copy(r.derived(), *this);
		}

		array1d& operator = (const array1d& r)
		{
			if (this != &r)
			{
				m_block = r.m_block;
			}
			return *this;
		}

		void swap(array1d& r)
		{
			m_block.swap(r.m_block);
		}

		caview1d<T> cview() const
		{
			return caview1d<T>(pbase(), nelems());
		}

		aview1d<T> view()
		{
			return aview1d<T>(pbase(), nelems());
		}

	public:
		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return m_block.size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_block.nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nelems() == 0;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(nelems());
		}

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

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return m_block[i];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return m_block[i];
		}

		template<class IndexSelector>
		typename _detail::subview_helper1d<value_type, IndexSelector>::cview_type
		V(const IndexSelector& I) const
		{
			return csubview(*this, I);
		}

		template<class IndexSelector>
		typename _detail::subview_helper1d<value_type, IndexSelector>::view_type
		V(const IndexSelector& I)
		{
			return subview(*this, I);
		}

	private:
		block_type m_block;

	}; // end class array1d


	template<typename T, class Alloc>
	inline void swap(array1d<T, Alloc>& lhs, array1d<T, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	template<class Derived, typename T>
	inline array1d<index_t> find(const IConstRegularAView1D<Derived, T>& B)
	{
		index_t n = B.nelems();

		const Derived& Bd = B.derived();

		// count

		index_t c = 0;
		for (index_t i = 0; i < n; ++i)
		{
			if (Bd(i)) ++c;
		}
		array1d<index_t> r(c);

		// extract

		index_t k = 0;
		for(index_t i = 0; k < c; ++i)
		{
			if (Bd(i)) r[k++] = i;
		}

		return r;
	}


	// select elements from 1D array

	template<class Derived, typename T, class IndexSelector>
	inline array1d<T> select_elems(const IConstRegularAView1D<Derived, T>& a, const IndexSelector& inds)
	{
		const Derived& ad = a.derived();

		index_t n = (index_t)inds.size();
		array1d<T> r(n);

		T *pd = r.pbase();
		for (index_t i = 0; i < n; ++i)
		{
			pd[i] = ad(inds[i]);
		}

		return r;
	}

}

#endif 

