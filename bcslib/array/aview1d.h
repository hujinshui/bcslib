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

		static const bool is_dense = true;
		static const bool is_continuous = false;
		static const bool is_const_view = true;
	};

	template<typename T, class TIndexer>
	class caview1d_ex : public dense_caview1d_base<caview1d_ex<T, TIndexer> >
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

		static const bool is_dense = true;
		static const bool is_continuous = false;
		static const bool is_const_view = false;
	};


	template<typename T, class TIndexer>
	class aview1d_ex : public dense_aview1d_base<aview1d_ex<T, TIndexer> >
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
	 *  Continuous views
	 *
	 ******************************************************/

	// sub view extraction

	namespace _detail
	{
		template<typename T, class IndexSelector> struct subview_helper1d;
	}

	template<class Derived, class IndexSelector>
	inline typename _detail::subview_helper1d<typename Derived::value_type, IndexSelector>::cview_type
	csubview(const continuous_caview1d_base<Derived>& a, const IndexSelector& I)
	{
		return _detail::subview_helper1d<typename Derived::value_type, IndexSelector>::cview(
				a.pbase(), a.dim0(), I);
	}

	template<class Derived, class IndexSelector>
	inline typename _detail::subview_helper1d<typename Derived::value_type, IndexSelector>::view_type
	subview(continuous_aview1d_base<Derived>& a, const IndexSelector& I)
	{
		return _detail::subview_helper1d<typename Derived::value_type, IndexSelector>::view(
				a.pbase(), a.dim0(), I);
	}


	// classes

	template<typename T>
	struct aview_traits<caview1d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		static const bool is_dense = true;
		static const bool is_continuous = true;
		static const bool is_const_view = true;
	};

	template<typename T>
	class caview1d : public continuous_caview1d_base<caview1d<T> >
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
		typename _detail::subview_helper1d<value_type, IndexSelector>::cview_type
		V(const IndexSelector& I) const
		{
			return csubview(*this, I);
		}

	private:
		pointer m_pbase;
		index_type m_d0;

	}; // end class caview1d


	template<typename T>
	struct aview_traits<aview1d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		static const bool is_dense = true;
		static const bool is_continuous = true;
		static const bool is_const_view = false;
	};

	template<typename T>
	class aview1d : public continuous_aview1d_base<aview1d<T> >
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
		pointer m_pbase;
		index_type m_d0;

	}; // end class aview1d


	/************************************
	 *
	 * sub-view helper implementation
	 *
	 ************************************/

	namespace _detail
	{
		template<typename T, class IndexSelector>
		struct subview_helper1d
		{
			typedef typename indexer_map<IndexSelector>::type sub_indexer_type;

			typedef caview1d_ex<T, sub_indexer_type> cview_type;
			typedef aview1d_ex<T, sub_indexer_type> view_type;

			static cview_type cview(const T* pbase, index_t d0, const IndexSelector& I)
			{
				index_t offset = indexer_map<IndexSelector>::get_offset(d0, I);
				return cview_type(pbase + offset,
						indexer_map<IndexSelector>::get_indexer(d0, I));
			}

			static view_type view(T *pbase, index_t d0, const IndexSelector& I)
			{
				index_t offset = indexer_map<IndexSelector>::get_offset(d0, I);
				return view_type(pbase + offset,
						indexer_map<IndexSelector>::get_indexer(d0, I));
			}
		};


		template<typename T>
		struct subview_helper1d<T, whole>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T> view_type;

			static cview_type cview(const T* pbase, index_t d0, whole)
			{
				return cview_type(pbase, d0);
			}

			static view_type view(T *pbase, index_t d0, whole)
			{
				return view_type(pbase, d0);
			}
		};


		template<typename T>
		struct subview_helper1d<T, range>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T> view_type;

			static cview_type cview(const T* pbase, index_t d0, const range &rgn)
			{
				return cview_type(pbase + rgn.begin_index(), rgn.dim());
			}

			static view_type view(T *pbase, index_t d0, const range &rgn)
			{
				return view_type(pbase + rgn.begin_index(), rgn.dim());
			}
		};

	}


	/************************************
	 *
	 *  Convenient functions
	 *
	 ************************************/

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

	template<class Derived>
	inline caview1d<typename Derived::value_type> flatten(const continuous_caview_base<Derived>& a)
	{
		return caview1d<typename Derived::value_type>(a.pbase(), a.nelems());
	}

	template<class Derived>
	inline aview1d<typename Derived::value_type> flatten(continuous_aview_base<Derived>& a)
	{
		return caview1d<typename Derived::value_type>(a.pbase(), a.nelems());
	}
}

#define BCS_AVIEW_FLATTEN_TYPEDEFS(T) \
	typedef caview1d<T> flatten_cview_type; \
	typedef aview1d<T> flatten_view_type;


#endif
