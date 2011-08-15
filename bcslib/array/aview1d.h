/**
 * @file aview1d.h
 *
 * The classes for one-dimensional array views
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_AVIEW1D_H_
#define BCSLIB_AVIEW1D_H_

#include <bcslib/array/aview1d_base.h>

namespace bcs
{
	/******************************************************
	 *
	 *  Extended views
	 *
	 ******************************************************/

	template<typename T, class TIndexer>
	struct aview_traits<caview1d_ex<T, TIndexer> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef caview1d_ex<T, TIndexer> self_type;
		typedef caview1d_base<self_type> view_nd_base;
		typedef caview_base<self_type> dview_base;
		typedef caview_base<self_type> view_base;
	};

	template<typename T, class TIndexer>
	class caview1d_ex : public caview1d_base<caview1d_ex<T, TIndexer> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer> );

		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)
		typedef TIndexer indexer_type;

	public:
		caview1d_ex(const_pointer pbase, const indexer_type& indexer)
		: m_pbase(pbase), m_d0(indexer.dim()), m_indexer(indexer)
		{
		}

	public:

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return m_pbase[m_indexer[i]];
		}

	private:
		const_pointer m_pbase;
		index_type m_d0;
		indexer_type m_indexer;

	}; // end class caview1d_ex


	template<typename T, class TIndexer>
	struct aview_traits<aview1d_ex<T, TIndexer> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef aview1d_ex<T, TIndexer> self_type;
		typedef aview1d_base<self_type> view_nd_base;
		typedef aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};


	template<typename T, class TIndexer>
	class aview1d_ex : public aview1d_base<aview1d_ex<T, TIndexer> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer> );

		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef TIndexer indexer_type;

	public:
		aview1d_ex(pointer pbase, const indexer_type& indexer)
		: m_pbase(pbase), m_d0(indexer.dim()), m_indexer(indexer)
		{
		}

		operator caview1d_ex<T, TIndexer>() const
		{
			return caview1d_ex<T, TIndexer>(m_pbase, m_indexer);
		}

	public:

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_t i) const
		{
			return m_pbase[m_indexer[i]];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return m_pbase[m_indexer[i]];
		}

	private:
		pointer m_pbase;
		index_type m_d0;
		indexer_type m_indexer;

	}; // end class aview1d_ex


	/******************************************************
	 *
	 *  Dense views
	 *
	 ******************************************************/

	// sub view extraction

	template<class Derived, class IndexSelector>
	inline caview1d_ex<typename Derived::value_type, typename indexer_map<IndexSelector>::type>
	subview(const dense_caview1d_base<Derived>& a, const IndexSelector& I)
	{
		typedef caview1d_ex<typename Derived::value_type, typename indexer_map<IndexSelector>::type> ret_type;

		index_t offset = indexer_map<IndexSelector>::get_offset(a.dim0(), I);
		return ret_type(a.pbase() + offset,
				indexer_map<IndexSelector>::get_indexer(a.dim0(), I));
	}


	template<class Derived, class IndexSelector>
	inline aview1d_ex<typename Derived::value_type, typename indexer_map<IndexSelector>::type>
	subview(dense_aview1d_base<Derived>& a, const IndexSelector& I)
	{
		typedef aview1d_ex<typename Derived::value_type, typename indexer_map<IndexSelector>::type> ret_type;

		index_t offset = indexer_map<IndexSelector>::get_offset(a.dim0(), I);
		return ret_type(a.pbase() + offset,
				indexer_map<IndexSelector>::get_indexer(a.dim0(), I));
	}


	// classes

	template<typename T>
	struct aview_traits<caview1d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef caview1d<T> self_type;
		typedef caview1d_base<self_type> view_nd_base;
		typedef dense_caview_base<self_type> dview_base;
		typedef caview_base<self_type> view_base;
	};

	template<typename T>
	class caview1d : public dense_caview1d_base<caview1d<T> >
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

	public:
		caview1d(const_pointer pbase, index_type n)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(n)
		{
		}

		caview1d(const_pointer pbase, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase)), m_d0(shape[0])
		{
		}

	public:

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return m_pbase[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return m_pbase[i];
		}

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return subview(*this, I);
		}

	private:
		pointer m_pbase;
		index_type m_d0;

	}; // end class caview1d


	template<typename T>
	struct aview_traits<aview1d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef aview1d<T> self_type;
		typedef aview1d_base<self_type> view_nd_base;
		typedef dense_aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};

	template<typename T>
	class aview1d : public dense_aview1d_base<aview1d<T> >
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

	public:
		aview1d(pointer pbase, index_type n)
		: m_pbase(pbase), m_d0(n)
		{
		}

		aview1d(pointer pbase, const shape_type& shape)
		: m_pbase(pbase), m_d0(shape[0])
		{
		}

		operator caview1d<T>() const
		{
			return caview1d<T>(pbase(), dim0());
		}

	public:

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return m_pbase[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return m_pbase[i];
		}

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return subview(*this, I);
		}

		template<class IndexSelector>
		aview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I)
		{
			return subview(*this, I);
		}

	private:
		pointer m_pbase;
		index_type m_d0;

	}; // end class aview1d


	// convenient functions

	template<typename T>
	inline caview1d<T> make_caview1d(const T* data, index_t n)
	{
		return caview1d<T>(data, n);
	}

	template<typename T>
	inline aview1d<T> make_aview1d(T* data, index_t n)
	{
		return aview1d<T>(data, n);
	}

	template<class LDerived, class RDerived>
	inline bool is_same_shape(const caview1d_base<LDerived>& lhs, const caview1d_base<RDerived>& rhs)
	{
		return lhs.dim0() == rhs.dim0();
	}

	template<class Derived>
	inline caview1d<typename Derived::value_type> flatten(const dense_caview_base<Derived>& a)
	{
		return caview1d<typename Derived::value_type>(a.pbase(), a.nelems());
	}

	template<class Derived>
	inline aview1d<typename Derived::value_type> flatten(dense_aview_base<Derived>& a)
	{
		return caview1d<typename Derived::value_type>(a.pbase(), a.nelems());
	}
}

#endif
