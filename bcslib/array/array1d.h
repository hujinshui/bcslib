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
#include <bcslib/array/array_storage.h>

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
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

		void export_to(pointer dst) const
		{
			index_t d0 = dim0();
			for (index_type i = 0; i < d0; ++i)
			{
				*(dst++) = operator()(i);
			}
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
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

		void export_to(pointer dst) const
		{
			index_t d0 = dim0();
			for (index_type i = 0; i < d0; ++i)
			{
				*(dst++) = operator()(i);
			}
		}

		void import_from(const_pointer src)
		{
			index_t d0 = dim0();
			for (index_type i = 0; i < d0; ++i)
			{
				operator()(i) = *(src++);
			}
		}

		void fill(const value_type& v)
		{
			index_t d0 = dim0();
			for (index_type i = 0; i < d0; ++i)
			{
				operator()(i) = v;
			}
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
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

		void export_to(pointer dst) const
		{
			copy_elements(pbase(), dst, size());
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
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

		void export_to(pointer dst) const
		{
			copy_elements(pbase(), dst, size());
		}

		void import_from(const_pointer src)
		{
			copy_elements(src, pbase(), size());
		}

		void fill(const value_type& v)
		{
			fill_elements(pbase(), size(), v);
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




	/******************************************************
	 *
	 *  View operations
	 *
	 ******************************************************/

	// comparison

	template<class LDerived, class RDerived>
	inline bool is_same_shape(const caview1d_base<LDerived>& lhs, const caview1d_base<RDerived>& rhs)
	{
		return lhs.dim0() == rhs.dim0();
	}

	template<class LDerived, class RDerived>
	inline bool is_equal(const dense_caview1d_base<LDerived>& lhs, const dense_caview1d_base<RDerived>& rhs)
	{
		return is_same_shape(lhs, rhs) && elements_equal(lhs.pbase(), rhs.pbase(), lhs.size());
	}

	// copy

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview1d_base<LDerived>& src, dense_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.size());
	}

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview1d_base<LDerived>& src, aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		dst.import_from(src.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview1d_base<LDerived>& src, dense_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		src.export_to(dst.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview1d_base<LDerived>& src, aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");

		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		index_t d0 = src.dim0();
		for (index_t i = 0; i < d0; ++i)
		{
			dstd(i) = srcd(i);
		}
	}


	/******************************************************
	 *
	 *  stand-alone array class
	 *
	 ******************************************************/

	template<typename T, class Alloc>
	struct aview_traits<array1d<T, Alloc> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef array1d<T, Alloc> self_type;
		typedef aview1d_base<self_type> view_nd_base;
		typedef dense_aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};

	template<typename T, class Alloc>
	class array1d
	: public dense_aview1d_base<array1d<T, Alloc> >
	, private sharable_storage_base<T, Alloc>
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef sharable_storage_base<T, Alloc> storage_base;
		typedef aview1d<T> view_type;

	public:
		explicit array1d(index_type n)
		: storage_base((size_t)n), m_view(storage_base::pointer_to_base(), n)
		{
		}

		explicit array1d(const shape_type& shape)
		: storage_base((size_t)(shape[0])), m_view(storage_base::pointer_to_base(), shape[0])
		{
		}

		array1d(index_type n, const value_type& x)
		: storage_base((size_t)n, x), m_view(storage_base::pointer_to_base(), n)
		{
		}

		array1d(index_type n, const_pointer src)
		: storage_base((size_t)n, src), m_view(storage_base::pointer_to_base(), n)
		{
		}

		array1d(const array1d& r)
		: storage_base(r), m_view(storage_base::pointer_to_base(), r.nelems())
		{
		}

		array1d(array1d&& r)
		: storage_base(std::move(r)), m_view(std::move(r.m_view))
		{
			r.m_view = view_type(BCS_NULL, 0);
		}

		template<class Derived>
		explicit array1d(const caview1d_base<Derived>& r)
		: storage_base(r.size()), m_view(storage_base::pointer_to_base(), r.nelems())
		{
			copy(r.derived(), *this);
		}

		array1d(const array1d& r, do_share ds)
		: storage_base(r, ds), m_view(storage_base::pointer_to_base(), r.nelems())
		{

		}

		array1d shared_copy() const
		{
			return array1d(*this, do_share());
		}

		array1d& operator = (const array1d& r)
		{
			if (this != &r)
			{
				storage_base &s = *this;
				s = r;

				m_view = view_type(s.pointer_to_base(), r.nelems());
			}
			return *this;
		}

		array1d& operator = (array1d&& r)
		{
			storage_base &s = *this;
			s = std::move(r);

			m_view = std::move(r);
			r.m_view = view_type(BCS_NULL, 0);

			return *this;
		}

		void swap(array1d& r)
		{
			using std::swap;

			storage_base::swap(r);
			swap(m_view, r.m_view);
		}

		bool is_unique() const
		{
			return storage_base::is_unique();
		}

		void make_unique()
		{
			storage_base::make_unique();

			index_t n = dim0();
			m_view = view_type(storage_base::pointer_to_base(), n);
		}

		operator caview1d<T>() const
		{
			return caview1d<T>(pbase(), dim0());
		}

		operator aview1d<T>()
		{
			return aview1d<T>(pbase(), dim0());
		}

	public:
		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return m_view.size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_view.nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_view.is_empty();
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_view.dim0();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return m_view.shape();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_view.pbase();
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return m_view.pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return m_view[i];
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return m_view[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return m_view[i];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return m_view[i];
		}

		void export_to(pointer dst) const
		{
			m_view.export_to(dst);
		}

		void import_from(const_pointer src)
		{
			m_view.import_from(src);
		}

		void fill(const value_type& v)
		{
			m_view.fill(v);
		}

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return m_view.V(I);
		}

		template<class IndexSelector>
		aview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I)
		{
			return m_view.V(I);
		}

	private:
		view_type m_view;

	}; // end class array1d


	template<typename T, class Alloc>
	inline void swap(array1d<T, Alloc>& lhs, array1d<T, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	template<class Derived>
	inline array1d<typename Derived::value_type> clone_array(const caview1d_base<Derived>& a)
	{
		return array1d<typename Derived::value_type>(a);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	template<class Derived>
	inline array1d<index_t> find(const caview1d_base<Derived>& B)
	{
		index_t n = B.dim0();

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

	template<class Derived, class IndexSelector>
	inline array1d<typename Derived::value_type>
	select_elems(const caview1d_base<Derived>& a, const IndexSelector& inds)
	{
		typedef typename Derived::value_type T;

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

