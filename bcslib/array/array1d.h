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


#include <bcslib/array/array_base.h>
#include <bcslib/array/array_index.h>
#include <bcslib/array/array_storage.h>

namespace bcs
{

	/******************************************************
	 *
	 *  Basic concepts for 1D
	 *
	 ******************************************************/

	template<class Derived>
	class caview1d_base : public aview_traits<Derived>::dview_base
	{
	public:
		BCS_CAVIEW_BASE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		// interfaces to be implemented by Derived

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return derived().shape();
		}

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		// -- new --

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

	}; // end class caview1d_base


	template<class Derived>
	class aview1d_base : public caview1d_base<Derived>
	{
	public:
		BCS_AVIEW_BASE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		// interfaces to be implemented by Derived

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return derived().shape();
		}

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		void import_from(const_pointer src)
		{
			derived().import_from(src);
		}

		void fill(const value_type& v)
		{
			derived().fill(v);
		}

		// -- new --

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_t i) const
		{
			return derived().operator()(i);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return derived().operator()(i);
		}

	}; // end class aview1d_base


	template<class Derived>
	class dense_caview1d_base : public aview_traits<Derived>::view_nd_base
	{
	public:
		BCS_CAVIEW_BASE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		// interfaces to be implemented by Derived

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return derived().shape();
		}

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return derived().operator[](i);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

		// -- new --

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return derived().V(I);
		}

	}; // end class dense_caview1d_base


	template<class Derived>
	class dense_aview1d_base : public dense_caview1d_base<Derived>
	{
	public:
		BCS_AVIEW_BASE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		// interfaces to be implemented by Derived

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return derived().shape();
		}

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		void import_from(const_pointer src)
		{
			derived().import_from(src);
		}

		void fill(const value_type& v)
		{
			derived().fill(v);
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return derived().operator[](i);
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return derived().operator[](i);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return derived().operator()(i);
		}

		// -- new --

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return derived().V(I);
		}

		template<class IndexSelector>
		aview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I)
		{
			return derived().V(I);
		}

	}; // end class dense_aview1d_base


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
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(indexer.dim())
		, m_indexer(indexer)
		{
		}

	public:

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
			for (index_type i = 0; i < m_d0; ++i)
			{
				*(dst++) = operator()(i);
			}
		}

	protected:
		pointer m_pbase;
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
	class aview1d_ex : public caview1d_ex<T, TIndexer>, public aview1d_base<aview1d_ex<T, TIndexer> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer> );

		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef caview1d_ex<T, TIndexer> super;
		typedef TIndexer indexer_type;

	public:
		aview1d_ex(pointer pbase, const indexer_type& indexer)
		: super(pbase, indexer)
		{
		}

	public:
		BCS_ENSURE_INLINE size_type size() const
		{
			return super::size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return super::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return super::is_empty();
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return super::dim0();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return super::shape();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_t i) const
		{
			return this->m_pbase[this->m_indexer[i]];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return this->m_pbase[this->m_indexer[i]];
		}

		void export_to(pointer dst) const
		{
			super::export_to(dst);
		}

		void import_from(const_pointer src)
		{
			for (index_type i = 0; i < this->m_d0; ++i)
			{
				operator()(i) = *(src++);
			}
		}

		void fill(const value_type& v)
		{
			for (index_type i = 0; i < this->m_d0; ++i)
			{
				operator()(i) = v;
			}
		}

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

	protected:
		void reset()
		{
			m_pbase = BCS_NULL;
			m_d0 = 0;
		}

	protected:
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
	class aview1d : public caview1d<T>, public dense_aview1d_base<aview1d<T> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)
		typedef caview1d<T> super;

	public:
		aview1d(pointer pbase, index_type n)
		: super(pbase, n)
		{
		}

		aview1d(pointer pbase, const shape_type& shape)
		: super(pbase, shape)
		{
		}

	public:
		BCS_ENSURE_INLINE size_type size() const
		{
			return super::size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return super::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return super::is_empty();
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return super::dim0();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return super::shape();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return this->m_pbase;
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return this->m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return this->m_pbase[i];
		}

		void export_to(pointer dst) const
		{
			super::export_to(dst);
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
		check_arg(src.dim0() == dst.dim0(), "aview1d copy: the shapes of src and dst are inconsistent.");
		src.export_to(dst.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview1d_base<LDerived>& src, aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");

		index_t d0 = src.dim0();
		for (index_t i = 0; i < d0; ++i)
		{
			dst(i) = src(i);
		}
	}


	/******************************************************
	 *
	 *  stand-alone array class
	 *
	 ******************************************************/

	template<typename T>
	struct aview_traits<array1d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef array1d<T> self_type;
		typedef aview1d_base<self_type> view_nd_base;
		typedef dense_aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};

	template<typename T, class Alloc>
	class array1d :
		private sharable_storage_base<T, Alloc>, private aview1d<T>, public dense_aview1d_base<array1d<T> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
		BCS_AVIEW_TRAITS_DEFS(1u, T, layout_1d_t)

		typedef sharable_storage_base<T, Alloc> storage_base;
		typedef aview1d<T> view_base;

	public:
		explicit array1d(index_type n)
		: storage_base((size_t)n), view_base(storage_base::pointer_to_base(), n)
		{
		}

		explicit array1d(const shape_type& shape)
		: storage_base((size_t)(shape[0])), view_base(storage_base::pointer_to_base(), shape[0])
		{
		}

		array1d(index_type n, const value_type& x)
		: storage_base((size_t)n, x), view_base(storage_base::pointer_to_base(), n)
		{
		}

		array1d(index_type n, const_pointer src)
		: storage_base((size_t)n, src), view_base(storage_base::pointer_to_base(), n)
		{
		}

		array1d(const array1d& r)
		: storage_base(r), view_base(storage_base::pointer_to_base(), r.nelems())
		{
		}

		array1d(array1d&& r)
		: storage_base(std::move(r)), view_base(std::move(r))
		{
			view_base& rv = r;
			rv.reset();
		}

		template<class Derived>
		explicit array1d(const caview1d_base<Derived>& r)
		: storage_base(r.size()), view_base(storage_base::pointer_to_base(), r.nelems())
		{
			copy(r, *this);
		}

		array1d(const array1d& r, do_share ds)
		: storage_base(r, ds), view_base(storage_base::pointer_to_base(), r.nelems())
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
				view_base& v = *this;

				s = r;
				v = view_base(s.pointer_to_base(), r.nelems());
			}
			return *this;
		}

		array1d& operator = (array1d&& r)
		{
			storage_base &s = *this;
			view_base& v = *this;

			s = std::move(r);
			v = std::move(r);

			view_base& rv = r;
			rv.reset();

			return *this;
		}

		void swap(array1d& r)
		{
			using std::swap;

			storage_base::swap(r);

			view_base& v = *this;
			view_base& rv = r;
			swap(v, rv);
		}

		bool is_unique() const
		{
			return storage_base::is_unique();
		}

		void make_unique()
		{
			storage_base::make_unique();

			view_base& v = *this;
			index_t n = v.dim0();
			v = view_base(storage_base::pointer_to_base(), n);
		}

	public:
		BCS_ENSURE_INLINE size_type size() const
		{
			return view_base::size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return view_base::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return view_base::is_empty();
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return view_base::dim0();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return view_base::shape();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return view_base::pbase();
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return view_base::pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator[](index_type i)
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return this->m_pbase[i];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return this->m_pbase[i];
		}

		void export_to(pointer dst) const
		{
			view_base::export_to(dst);
		}

		void import_from(const_pointer src)
		{
			view_base::import_from(src);
		}

		void fill(const value_type& v)
		{
			view_base::fill(v);
		}

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return view_base::V(I);
		}

		template<class IndexSelector>
		aview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I)
		{
			return view_base::V(I);
		}

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

		// count

		index_t c = 0;
		for (index_t i = 0; i < n; ++i)
		{
			if (B(i)) ++c;
		}
		array1d<index_t> r(c);

		// extract

		index_t k = 0;
		for(index_t i = 0; k < c; ++i)
		{
			if (B(i)) r[k++] = i;
		}

		return r;
	}


	// select elements from 1D array

	template<class Derived, class IndexSelector>
	inline array1d<typename Derived::value_type>
	select_elems(const caview1d_base<Derived>& a, const IndexSelector& inds)
	{
		typedef typename Derived::value_type T;

		index_t n = (index_t)inds.size();
		array1d<T> r(n);

		T *pd = r.pbase();
		for (index_t i = 0; i < n; ++i)
		{
			pd[i] = a(inds[i]);
		}

		return r;
	}

}

#endif 

