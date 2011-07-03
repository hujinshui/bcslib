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
#include <bcslib/base/block.h>

namespace bcs
{
	/******************************************************
	 *
	 *  Extended views
	 *
	 ******************************************************/

	template<typename T, class TIndexer>
	class caview1d_ex
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer> );
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef TIndexer indexer_type;
		typedef caview1d_ex<value_type, indexer_type> cview_type;
		typedef aview1d_ex<value_type, indexer_type> view_type;

	public:
		caview1d_ex(const_pointer pbase, const indexer_type& indexer)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(static_cast<index_t>(indexer.size()))
		, m_indexer(indexer)
		{
		}

	private:
		caview1d_ex& operator = (const caview1d_ex& r);

	public:
		dim_num_t ndims() const
		{
			return num_dimensions;
		}

		size_type size() const
		{
			return nelems();
		}

		size_type nelems() const
		{
			return static_cast<size_type>(m_d0);
		}

		index_type dim0() const
		{
			return m_d0;
		}

		shape_type shape() const
		{
			return arr_shape(m_d0);
		}

		const indexer_type& get_indexer() const
		{
			return m_indexer;
		}

	public:
		// Element access

		const_reference operator() (index_type i) const
		{
			return m_pbase[m_indexer[i]];
		}

	protected:
		pointer m_pbase;
		index_type m_d0;
		indexer_type m_indexer;

	}; // end class caview1d_ex



	template<typename T, class TIndexer>
	class aview1d_ex : public caview1d_ex<T, TIndexer>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer> );
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef TIndexer indexer_type;
		typedef caview1d_ex<value_type, indexer_type> cview_type;
		typedef aview1d_ex<value_type, indexer_type> view_type;

	public:
		aview1d_ex(pointer pbase, const indexer_type& indexer)
		: cview_type(pbase, indexer)
		{
		}

		aview1d_ex(aview1d_ex& r)
		: cview_type(r)
		{
		}

		aview1d_ex(aview1d_ex&& r)
		: cview_type(r)
		{
		}

	private:
		aview1d_ex& operator = (const aview1d_ex& r);

	public:
		// Element access

		const_reference operator() (index_type i) const
		{
			return this->m_pbase[this->m_indexer[i]];
		}

		reference operator() (index_type i)
		{
			return this->m_pbase[this->m_indexer[i]];
		}

	}; // end class aview1d_ex


	/******************************************************
	 *
	 *  Dense views
	 *
	 ******************************************************/

	template<typename T>
	class caview1d
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef caview1d<value_type> cview_type;
		typedef aview1d<value_type> view_type;

		typedef const_pointer const_iterator;
		typedef pointer iterator;

	public:
		caview1d(const_pointer pbase, size_type n)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(static_cast<index_t>(n))
		{
		}

		caview1d(const_pointer pbase, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase)), m_d0(shape[0])
		{
		}

		caview1d(const caview1d& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0)
		{
		}

		caview1d(caview1d&& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0)
		{
			r.reset();
		}

		caview1d& operator = (const caview1d& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			return *this;
		}

		caview1d& operator = (caview1d&& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			r.reset();
			return *this;
		}

	public:
		dim_num_t ndims() const
		{
			return num_dimensions;
		}

		size_type size() const
		{
			return nelems();
		}

		size_type nelems() const
		{
			return static_cast<size_type>(m_d0);
		}

		index_type dim0() const
		{
			return m_d0;
		}

		shape_type shape() const
		{
			return arr_shape(m_d0);
		}

	public:
		// Element access

		const_pointer pbase() const
		{
			return m_pbase;
		}

		const_pointer ptr(index_type i) const
		{
			return m_pbase + i;
		}

		const_reference operator[] (index_type i) const
		{
			return m_pbase[i];
		}

		const_reference operator() (index_type i) const
		{
			return m_pbase[i];
		}

		// Iteration

		const_iterator begin() const
		{
			return m_pbase;
		}

		const_iterator end() const
		{
			return m_pbase + m_d0;
		}

	public:
		// Sub-view

		cview_type V(whole) const
		{
			return *this;
		}

		cview_type V(const range& rgn) const
		{
			return cview_type(m_pbase + rgn.begin_index(), rgn.size());
		}

		template<class Indexer>
		caview1d_ex<value_type, typename indexer_remap<Indexer>::type> V(const Indexer& indexer) const
		{
			return caview1d_ex<value_type, Indexer>(m_pbase, indexer_remap<Indexer>::get(indexer));
		}

	protected:
		pointer m_pbase;
		index_type m_d0;

	private:
		void reset()
		{
			m_pbase = BCS_NULL;
			m_d0 = 0;
		}

	}; // end class caview1d


	template<typename T>
	class aview1d : public caview1d<T>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef caview1d<value_type> cview_type;
		typedef aview1d<value_type> view_type;

		typedef const_pointer const_iterator;
		typedef pointer iterator;

	public:
		aview1d(pointer pbase, size_type n)
		: cview_type(pbase, n)
		{
		}

		aview1d(pointer pbase, const shape_type& shape)
		: cview_type(pbase, shape)
		{
		}

		aview1d(aview1d& r)
		: cview_type(r)
		{
		}

		aview1d(aview1d&& r)
		: cview_type(std::move(r))
		{
		}

		aview1d& operator = (aview1d& r)
		{
			cview_type::operator =(r);
			return *this;
		}

		aview1d& operator = (aview1d&& r)
		{
			cview_type::operator = (std::move(r));
			return *this;
		}

	public:
		// Element access

		const_pointer pbase() const
		{
			return this->m_pbase;
		}

		pointer pbase()
		{
			return this->m_pbase;
		}

		const_pointer ptr(index_type i) const
		{
			return this->m_pbase + i;
		}

		pointer ptr(index_type i)
		{
			return this->m_pbase + i;
		}

		const_reference operator[] (index_type i) const
		{
			return this->m_pbase[i];
		}

		reference operator[] (index_type i)
		{
			return this->m_pbase[i];
		}

		const_reference operator() (index_type i) const
		{
			return this->m_pbase[i];
		}

		reference operator() (index_type i)
		{
			return this->m_pbase[i];
		}

		// Iteration

		const_iterator begin() const
		{
			return this->m_pbase;
		}

		iterator begin()
		{
			return this->m_pbase;
		}

		const_iterator end() const
		{
			return this->m_pbase + this->m_d0;
		}

		iterator end()
		{
			return this->m_pbase + this->m_d0;
		}

	public:
		// Sub-view

		cview_type V(const range& rgn) const
		{
			return cview_type(this->m_pbase + rgn.begin_index(), rgn.size());
		}

		view_type V(const range& rgn)
		{
			return view_type(this->m_pbase + rgn.begin_index(), rgn.size());
		}

		template<class Indexer>
		caview1d_ex<value_type, Indexer> V(const Indexer& indexer) const
		{
			return caview1d_ex<value_type, Indexer>(this->m_pbase, indexer);
		}

		template<class Indexer>
		aview1d_ex<value_type, Indexer> V(const Indexer& indexer)
		{
			return aview1d_ex<value_type, Indexer>(this->m_pbase, indexer);
		}

		cview_type V(whole) const
		{
			return *this;
		}

		view_type V(whole)
		{
			return *this;
		}

		caview1d_ex<value_type, step_range> V(rev_whole) const
		{
			return V(rgn(this->m_d0, rev_whole()));
		}

		aview1d_ex<value_type, step_range> V(rev_whole)
		{
			return V(rgn(this->m_d0, rev_whole()));
		}

	}; // end class caview1d


	// iteration functions

	template<typename T>
	typename caview1d<T>::const_iterator begin(const caview1d<T>& a)
	{
		return a.begin();
	}

	template<typename T>
	typename caview1d<T>::const_iterator end(const caview1d<T>& a)
	{
		return a.end();
	}

	template<typename T>
	typename aview1d<T>::iterator begin(aview1d<T>& a)
	{
		return a.begin();
	}

	template<typename T>
	typename aview1d<T>::iterator end(aview1d<T>& a)
	{
		return a.end();
	}


	/******************************************************
	 *
	 *  View operations
	 *
	 ******************************************************/

	// element-wise comparison

	template<typename T>
	inline bool operator == (const caview1d<T>& lhs, const caview1d<T>& rhs)
	{
		return lhs.dim0() == rhs.dim0() && elements_equal(lhs.pbase(), rhs.pbase(), lhs.nelems());
	}

	template<typename T>
	inline bool operator != (const caview1d<T>& lhs, const caview1d<T>& rhs)
	{
		return !(lhs == rhs);
	}

	// export & import

	template<typename T>
	inline void import_from(aview1d<T>& a, const T *in)
	{
		copy_elements(in, a.pbase(), a.nelems());
	}

	template<typename T, typename ForwardIterator>
	inline void import_from(aview1d<T>& a, ForwardIterator in)
	{
		std::copy_n(in, a.nelems(), a.begin());
	}

	template<typename T, class TIndexer, typename ForwardIterator>
	inline void import_from(aview1d_ex<T, TIndexer>& a, ForwardIterator in)
	{
		index_t d0 = a.dim0();
		for (index_t i = 0; i < d0; ++i, ++in)
		{
			a(i) = *in;
		}
	}

	template<typename T>
	inline void export_to(const caview1d<T>& a, T *out)
	{
		copy_elements(a.pbase(), out, a.nelems());
	}

	template<typename T, typename ForwardIterator>
	inline void export_to(const caview1d<T>& a, ForwardIterator out)
	{
		std::copy_n(a.begin(), a.nelems(), out);
	}

	template<typename T, class TIndexer, typename ForwardIterator>
	inline void export_to(const caview1d_ex<T, TIndexer>& a, ForwardIterator out)
	{
		index_t d0 = a.dim0();
		for (index_t i = 0; i < d0; ++i, ++out)
		{
			*out = a(i);
		}
	}


	// fill

	template<typename T>
	inline void set_zeros(aview1d<T>& dst)
	{
		set_zeros_to_elements(dst.pbase(), dst.nelems());
	}

	template<typename T>
	inline void fill(aview1d<T>& dst, const T& v)
	{
		fill_elements(dst.pbase(), dst.nelems(), v);
	}

	template<typename T, typename TIndexer>
	inline void fill(aview1d_ex<T, TIndexer>& dst, const T& v)
	{
		index_t d0 = dst.dim0();
		for (index_t i = 0; i < d0; ++i)
		{
			dst(i) = v;
		}
	}


	// copy

	template<typename T>
	inline void copy(const caview1d<T>& src, aview1d<T>& dst)
	{
		check_arg(src.dim0() == dst.dim0(), "aview1d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.nelems());
	}

	template<typename T, class TIndexer>
	inline void copy(const caview1d<T>& src, aview1d_ex<T, TIndexer>& dst)
	{
		check_arg(src.dim0() == dst.dim0(), "aview1d copy: the shapes of src and dst are inconsistent.");
		import_from(dst, src.begin());
	}

	template<typename T, class TIndexer>
	inline void copy(const caview1d_ex<T, TIndexer>& src, aview1d<T>& dst)
	{
		check_arg(src.dim0() == dst.dim0(), "aview1d copy: the shapes of src and dst are inconsistent.");
		export_to(src, dst.begin());
	}

	template<typename T, class LIndexer, class RIndexer>
	inline void copy(const caview1d_ex<T, LIndexer>& src, aview1d_ex<T, RIndexer>& dst)
	{
		check_arg(src.dim0() == dst.dim0(), "aview1d copy: the shapes of src and dst are inconsistent.");

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

	template<typename T, class Alloc>
	class array1d : private sharable_storage_base<T, Alloc>, public aview1d<T>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_ARRAY_BASIC_TYPEDEFS(1u, T, layout_1d_t)

		typedef caview1d<value_type> cview_type;
		typedef aview1d<value_type> view_type;

		typedef typename view_type::const_iterator const_iterator;
		typedef typename view_type::iterator iterator;

		typedef sharable_storage_base<T, Alloc> storage_base_type;

	public:
		explicit array1d(size_type n)
		: storage_base_type(n), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		explicit array1d(const shape_type& shape)
		: storage_base_type(static_cast<size_type>(shape[0]))
		, view_type(storage_base_type::pointer_to_base(), (size_t)shape[0])
		{
		}

		array1d(size_type n, const value_type& x)
		: storage_base_type(n, x), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		array1d(size_type n, const_pointer src)
		: storage_base_type(n, src), view_type(storage_base_type::pointer_to_base(), n)
		{
		}

		array1d(const array1d& r)
		: storage_base_type(r), view_type(storage_base_type::pointer_to_base(), r.nelems())
		{
		}

		array1d(array1d&& r)
		: storage_base_type(std::move(r)), view_type(std::move(r))
		{
		}

		explicit array1d(const cview_type& r)
		: storage_base_type(r.nelems(), r.pbase())
		, view_type(storage_base_type::pointer_to_base(), r.nelems())
		{
		}

		template<typename Indexer>
		explicit array1d(const caview1d_ex<value_type, Indexer>& r)
		: storage_base_type(r.nelems()), view_type(storage_base_type::pointer_to_base(), r.nelems())
		{
			copy(r, *this);
		}

		array1d& operator = (const array1d& r)
		{
			if (this != &r)
			{
				storage_base_type &s = *this;
				view_type& v = *this;

				s = r;
				v = view_type(s.pointer_to_base(), r.nelems());
			}
			return *this;
		}

		array1d& operator = (array1d&& r)
		{
			storage_base_type &s = *this;
			view_type& v = *this;

			s = std::move(r);
			v = std::move(r);

			return *this;
		}

		void swap(array1d& r)
		{
			using std::swap;

			storage_base_type::swap(r);

			view_type& v = *this;
			view_type& rv = r;
			swap(v, rv);
		}

	public:
		// sharing

		array1d(const array1d& r, do_share ds)
		: storage_base_type(r, ds), view_type(storage_base_type::pointer_to_base(), r.nelems())
		{

		}

		array1d shared_copy() const
		{
			return array1d(*this, do_share());
		}

	}; // end class array1d


	template<typename T, class Alloc>
	inline void swap(array1d<T, Alloc>& lhs, array1d<T, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}


	template<typename T>
	inline array1d<T> clone_array(const caview1d<T>& a)
	{
		return array1d<T>(a);
	}

	template<typename T, class Indexer>
	inline array1d<T> clone_array(const caview1d_ex<T, Indexer>& a)
	{
		return array1d<T>(a);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	inline array1d<index_t> find(const caview1d<bool>& B)
	{
		index_t n = B.dim0();

		// count

		index_t c = 0;
		for (index_t i = 0; i < n; ++i)
		{
			c += (index_t)B[i];
		}
		array1d<index_t> r((size_t)c);

		// extract

		index_t k = 0;
		for(index_t i = 0; k < c; ++i)
		{
			if (B[i]) r[k++] = i;
		}

		return r;
	}


	// select elements from 1D array

	template<typename T, class IndexSelector>
	inline array1d<T> select_elems(const caview1d<T>& a, const IndexSelector& inds)
	{
		size_t n = (size_t)inds.size();
		array1d<T> r(n);

		auto in = inds.begin();
		T *pd = r.pbase();
		while (n--)
		{
			*(pd++) = a[*in];
			++ in;
		}

		return r;
	}

}

#endif 

