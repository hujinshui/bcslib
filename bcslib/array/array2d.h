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

#include <bcslib/array/array1d.h>
#include <bcslib/array/details/array2d_details.h>

namespace bcs
{
	/******************************************************
	 *
	 *  A type to represent a matrix being transposed
	 *
	 ******************************************************/

	template<typename T, typename TOrd>
	class array2d_transposed_t
	{
	public:
		typedef T value_type;
		typedef caview2d<T, TOrd> cview_type;

		explicit array2d_transposed_t(const cview_type& a) : m_cref(a) { }

		array2d_transposed_t(const array2d_transposed_t& r) : m_cref(r.m_cref) { }

		const cview_type& get() const
		{
			return m_cref;
		}

	private:
		const cview_type& m_cref;
	};


	/******************************************************
	 *
	 *  Extended Views
	 *
	 ******************************************************/

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class caview2d_ex
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef caview2d_ex<value_type, layout_order, indexer0_type, indexer1_type> cview_type;
		typedef  aview2d_ex<value_type, layout_order, indexer0_type, indexer1_type> view_type;

	public:
		caview2d_ex(const_pointer pbase, index_t base_d0, index_t base_d1, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(const_cast<pointer>(pbase))
		, m_idxcore(base_d0, base_d1)
		, m_d0(static_cast<index_t>(indexer0.size()))
		, m_d1(static_cast<index_t>(indexer1.size()))
		, m_indexer0(indexer0)
		, m_indexer1(indexer1)
		{
		}

	private:
		caview2d_ex& operator = (const caview2d_ex& r);

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
			return static_cast<size_type>(m_d0 * m_d1);
		}

		size_type nrows() const
		{
			return static_cast<size_type>(m_d0);
		}

		size_type ncolumns() const
		{
			return static_cast<size_type>(m_d1);
		}

		index_type dim0() const
		{
			return m_d0;
		}

		index_type dim1() const
		{
			return m_d1;
		}

		shape_type shape() const
		{
			return arr_shape(m_d0, m_d1);
		}

		const indexer0_type& get_indexer0() const
		{
			return m_indexer0;
		}

		const indexer1_type& get_indexer1() const
		{
			return m_indexer1;
		}

	public:
		// Element access

		const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

	protected:
		index_type offset(index_type i, index_type j) const
		{
			return m_idxcore.offset(m_indexer0[i], m_indexer1[j]);
		}

		pointer m_pbase;
		_detail::index_core2d<layout_order> m_idxcore;
		index_t m_d0;
		index_t m_d1;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class caview2d_ex


	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d_ex : public caview2d_ex<T, TOrd, TIndexer0, TIndexer1>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;
		typedef caview2d_ex<value_type, layout_order, indexer0_type, indexer1_type> cview_type;
		typedef  aview2d_ex<value_type, layout_order, indexer0_type, indexer1_type> view_type;

	public:
		aview2d_ex(pointer pbase, index_t base_d0, index_t base_d1, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: cview_type(pbase, base_d0, base_d1, indexer0, indexer1)
		{
		}

		aview2d_ex(aview2d_ex& r)
		: cview_type(r)
		{
		}

		aview2d_ex(aview2d_ex&& r)
		: cview_type(r)
		{
		}

	private:
		aview2d_ex& operator = (const aview2d_ex& r);

	public:
		// Element access

		const_reference operator() (index_type i, index_type j) const
		{
			return this->m_pbase[this->offset(i, j)];
		}

		reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[this->offset(i, j)];
		}

	}; // end class aview2d_ex


	/******************************************************
	 *
	 *  Dense views
	 *
	 ******************************************************/

	template<typename T, typename TOrd>
	class caview2d
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef caview2d<value_type, layout_order> cview_type;
		typedef aview2d<value_type, layout_order> view_type;

	private:
		typedef _detail::index_core2d<layout_order> index_core_t;
		typedef _detail::iter_helper2d<value_type, layout_order> iter_helper;
		typedef _detail::slice_helper2d<value_type, layout_order> slice_helper;

	public:
		typedef typename iter_helper::const_iterator const_iterator;
		typedef typename iter_helper::iterator iterator;


	public:
		caview2d(const_pointer pbase, size_type m, size_type n)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(static_cast<index_type>(m))
		, m_d1(static_cast<index_type>(n))
		, m_idxcore(m_d0, m_d1)
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(shape[0]), m_d1(shape[1])
		, m_idxcore(m_d0, m_d1)
		{
		}

		caview2d(const_pointer pbase, size_type m, size_type n, index_type base_d0, index_type base_d1)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(static_cast<index_type>(m))
		, m_d1(static_cast<index_type>(n))
		, m_idxcore(base_d0, base_d1)
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape, index_type base_d0, index_type base_d1)
		: m_pbase(const_cast<pointer>(pbase))
		, m_d0(shape[0]), m_d1(shape[1])
		, m_idxcore(base_d0, base_d1)
		{
		}


		caview2d(const caview2d& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0), m_d1(r.m_d1)
		, m_idxcore(r.m_idxcore)
		{
		}

		caview2d(caview2d&& r)
		: m_pbase(r.m_pbase)
		, m_d0(r.m_d0), m_d1(r.m_d1)
		, m_idxcore(r.m_idxcore)
		{
			r.reset();
		}

		caview2d& operator = (const caview2d& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			m_d1 = r.m_d1;
			m_idxcore = r.m_idxcore;

			return *this;
		}

		caview2d& operator = (caview2d&& r)
		{
			m_pbase = r.m_pbase;
			m_d0 = r.m_d0;
			m_d1 = r.m_d1;
			m_idxcore = r.m_idxcore;

			r.reset();

			return *this;
		}

	public:
		dim_num_t ndims() const
		{
			return num_dimensions;
		}

		size_type nelems() const
		{
			return static_cast<size_t>(m_d0 * m_d1);
		}

		index_type dim0() const
		{
			return m_d0;
		}

		index_type dim1() const
		{
			return m_d1;
		}

		size_type nrows() const
		{
			return static_cast<size_type>(dim0());
		}

		size_type ncolumns() const
		{
			return static_cast<size_type>(dim1());
		}

		shape_type shape() const
		{
			return arr_shape(dim0(), dim1());
		}

		index_type base_dim0() const
		{
			return m_idxcore.base_dim0();
		}

		index_type base_dim1() const
		{
			return m_idxcore.base_dim1();
		}

		shape_type base_shape() const
		{
			return m_idxcore.base_shape();
		}

		slice2d_info slice_info() const
		{
			return m_idxcore.get_slice_info(nrows(), ncolumns());
		}

		// just for temporary use in an expression, not allowed to pass around
		array2d_transposed_t<T, TOrd> Tp() const
		{
			return array2d_transposed_t<T, TOrd>(*this);
		}

	public:
		// Element access

		const_pointer pbase() const
		{
			return m_pbase;
		}

		const_pointer ptr(index_type i, index_type j) const
		{
			return m_pbase + m_idxcore.offset(i, j);
		}

		const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

		// Iteration

		const_iterator begin() const
		{
			return iter_helper::get_const_begin(m_pbase, m_idxcore, m_d0, m_d1);
		}

		const_iterator end() const
		{
			return iter_helper::get_const_end(m_pbase, m_idxcore, m_d0, m_d1);
		}


	public:
		// Slice

		typename slice_helper::row_cview_type row(index_type i) const
		{
			return slice_helper::row_cview(m_idxcore, m_pbase + m_idxcore.row_offset(i), ncolumns());
		}

		typename slice_helper::column_cview_type column(index_type j) const
		{
			return slice_helper::column_cview(m_idxcore, m_pbase + m_idxcore.column_offset(j), nrows());
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_cview_type
		row(index_type i, const TIndexer& rgn) const
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_cview(
					m_idxcore, m_pbase + m_idxcore.row_offset(i), rgn);
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_cview_type
		column(index_t j, const TIndexer& rgn) const
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_cview(
					m_idxcore, m_pbase + m_idxcore.column_offset(j), rgn);
		}

		// Sub-view

		cview_type V(whole, whole) const
		{
			return *this;
		}

		cview_type V(const range& I, const range& J) const
		{
			return cview_type(ptr(I.begin_index(), J.begin_index()), I.size(), J.size(),
					base_dim0(), base_dim1());
		}

		cview_type V(const range& I, whole) const
		{
			return V(I, rgn(dim1(), whole()));
		}

		cview_type V(whole, const range& J) const
		{
			return V(rgn(dim0(), whole()), J);
		}


		template<class TIndexer0, class TIndexer1>
		caview2d_ex<T, TOrd, typename indexer_remap<TIndexer0>::type, typename indexer_remap<TIndexer1>::type>
		V(const TIndexer0& I, const TIndexer1& J) const
		{
			typedef typename indexer_remap<TIndexer0>::type indexer0_t;
			typedef typename indexer_remap<TIndexer1>::type indexer1_t;

			return caview2d_ex<T, TOrd, indexer0_t, indexer1_t>(m_pbase, base_dim0(), base_dim1(),
					indexer_remap<TIndexer0>::get(dim0(), I), indexer_remap<TIndexer1>::get(dim1(), J));
		}

	protected:
		pointer m_pbase;
		index_type m_d0;
		index_type m_d1;
		index_core_t m_idxcore;

	private:
		void reset()
		{
			m_pbase = BCS_NULL;
			m_d0 = 0;
			m_d1 = 0;
		}

	}; // end class caview2d



	template<typename T, typename TOrd>
	class aview2d : public caview2d<T, TOrd>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef caview2d<value_type, layout_order> cview_type;
		typedef aview2d<value_type, layout_order> view_type;

	private:
		typedef _detail::index_core2d<layout_order> index_core_t;
		typedef _detail::iter_helper2d<value_type, layout_order> iter_helper;
		typedef _detail::slice_helper2d<value_type, layout_order> slice_helper;

	public:
		typedef typename iter_helper::const_iterator const_iterator;
		typedef typename iter_helper::iterator iterator;

	public:
		aview2d(const_pointer pbase, size_type m, size_type n)
		: cview_type(pbase, m, n)
		{
		}

		aview2d(const_pointer pbase, const shape_type& shape)
		: cview_type(pbase, shape)
		{
		}

		aview2d(const_pointer pbase, size_type m, size_type n, index_type base_d0, index_type base_d1)
		: cview_type(pbase, m, n, base_d0, base_d1)
		{
		}

		aview2d(const_pointer pbase, const shape_type& shape, index_type base_d0, index_type base_d1)
		: cview_type(pbase, shape, base_d0, base_d1)
		{
		}

		aview2d(aview2d& r)
		: cview_type(r)
		{
		}

		aview2d(aview2d&& r)
		: cview_type(std::move(r))
		{
		}

		aview2d& operator = (aview2d& r)
		{
			cview_type::operator = (r);
			return *this;
		}

		aview2d& operator = (aview2d&& r)
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

		const_pointer ptr(index_type i, index_type j) const
		{
			return this->m_pbase + this->m_idxcore.offset(i, j);
		}

		pointer ptr(index_type i, index_type j)
		{
			return this->m_pbase + this->m_idxcore.offset(i, j);
		}

		const_reference operator() (index_type i, index_type j) const
		{
			return this->m_pbase[this->m_idxcore.offset(i, j)];
		}

		reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[this->m_idxcore.offset(i, j)];
		}

		// Iteration

		const_iterator begin() const
		{
			return iter_helper::get_const_begin(this->m_pbase, this->m_idxcore, this->m_d0, this->m_d1);
		}

		const_iterator end() const
		{
			return iter_helper::get_const_end(this->m_pbase, this->m_idxcore, this->m_d0, this->m_d1);
		}

		iterator begin()
		{
			return iter_helper::get_begin(this->m_pbase, this->m_idxcore, this->m_d0, this->m_d1);
		}

		iterator end()
		{
			return iter_helper::get_end(this->m_pbase, this->m_idxcore, this->m_d0, this->m_d1);
		}

	public:
		// Slice

		typename slice_helper::row_cview_type row(index_type i) const
		{
			return slice_helper::row_cview(this->m_idxcore, this->m_pbase + this->m_idxcore.row_offset(i), this->ncolumns());
		}

		typename slice_helper::column_cview_type column(index_type j) const
		{
			return slice_helper::column_cview(this->m_idxcore, this->m_pbase + this->m_idxcore.column_offset(j), this->nrows());
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_cview_type
		row(index_type i, const TIndexer& rgn) const
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_cview(
					this->m_idxcore, this->m_pbase + this->m_idxcore.row_offset(i), rgn);
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_cview_type
		column(index_t j, const TIndexer& rgn) const
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_cview(
					this->m_idxcore, this->m_pbase + this->m_idxcore.column_offset(j), rgn);
		}

		typename slice_helper::row_view_type row(index_type i)
		{
			return slice_helper::row_view(this->m_idxcore, this->m_pbase + this->m_idxcore.row_offset(i), this->ncolumns());
		}

		typename slice_helper::column_view_type column(index_type j)
		{
			return slice_helper::column_view(this->m_idxcore, this->m_pbase + this->m_idxcore.column_offset(j), this->nrows());
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_view_type
		row(index_type i, const TIndexer& rgn)
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::row_range_view(
					this->m_idxcore, this->m_pbase + this->m_idxcore.row_offset(i), rgn);
		}

		template<class TIndexer>
		typename _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_view_type
		column(index_t j, const TIndexer& rgn)
		{
			return _detail::slice_range_helper2d<value_type, layout_order, TIndexer>::column_range_view(
					this->m_idxcore, this->m_pbase + this->m_idxcore.column_offset(j), rgn);
		}


		// Sub-view

		cview_type V(whole, whole) const
		{
			return *this;
		}

		cview_type V(const range& I, const range& J) const
		{
			return cview_type(ptr(I.begin_index(), J.begin_index()), I.size(), J.size(),
					this->base_dim0(), this->base_dim1());
		}

		cview_type V(const range& I, whole) const
		{
			return V(I, rgn(this->dim1(), whole()));
		}

		cview_type V(whole, const range& J) const
		{
			return V(rgn(this->dim0(), whole()), J);
		}


		template<class TIndexer0, class TIndexer1>
		caview2d_ex<T, TOrd, typename indexer_remap<TIndexer0>::type, typename indexer_remap<TIndexer1>::type>
		V(const TIndexer0& I, const TIndexer1& J) const
		{
			typedef typename indexer_remap<TIndexer0>::type indexer0_t;
			typedef typename indexer_remap<TIndexer1>::type indexer1_t;

			return caview2d_ex<T, TOrd, indexer0_t, indexer1_t>(this->m_pbase, this->base_dim0(), this->base_dim1(),
					indexer_remap<TIndexer0>::get(this->dim0(), I), indexer_remap<TIndexer1>::get(this->dim1(), J));
		}


		view_type V(whole, whole)
		{
			return *this;
		}

		view_type V(const range& I, const range& J)
		{
			return view_type(ptr(I.begin_index(), J.begin_index()), I.size(), J.size(),
					this->base_dim0(), this->base_dim1());
		}

		view_type V(const range& I, whole)
		{
			return V(I, rgn(this->dim1(), whole()));
		}

		view_type V(whole, const range& J)
		{
			return V(rgn(this->dim0(), whole()), J);
		}


		template<class TIndexer0, class TIndexer1>
		aview2d_ex<T, TOrd, typename indexer_remap<TIndexer0>::type, typename indexer_remap<TIndexer1>::type>
		V(const TIndexer0& I, const TIndexer1& J)
		{
			typedef typename indexer_remap<TIndexer0>::type indexer0_t;
			typedef typename indexer_remap<TIndexer1>::type indexer1_t;

			return aview2d_ex<T, TOrd, indexer0_t, indexer1_t>(this->m_pbase, this->base_dim0(), this->base_dim1(),
					indexer_remap<TIndexer0>::get(this->dim0(), I), indexer_remap<TIndexer1>::get(this->dim1(), J));
		}


	}; // end class aview2d


	// convenient functions to make 2D view

	template<typename T>
	inline caview2d<T, row_major_t> get_aview2d_rm(const T *base, size_t m, size_t n)
	{
		return caview2d<T, row_major_t>(base, m, n);
	}

	template<typename T>
	inline aview2d<T, row_major_t> get_aview2d_rm(T *base, size_t m, size_t n)
	{
		return aview2d<T, row_major_t>(base, m, n);
	}

	template<typename T>
	inline caview2d<T, column_major_t> get_aview2d_cm(const T *base, size_t m, size_t n)
	{
		return caview2d<T, column_major_t>(base, m, n);
	}

	template<typename T>
	inline aview2d<T, column_major_t> get_aview2d_cm(T *base, size_t m, size_t n)
	{
		return aview2d<T, column_major_t>(base, m, n);
	}

	// iteration functions

	template<typename T, typename TOrd>
	typename caview2d<T, TOrd>::const_iterator begin(const caview2d<T, TOrd>& a)
	{
		return a.begin();
	}

	template<typename T, typename TOrd>
	typename caview2d<T, TOrd>::const_iterator end(const caview2d<T, TOrd>& a)
	{
		return a.end();
	}

	template<typename T, typename TOrd>
	typename aview2d<T, TOrd>::iterator begin(aview2d<T, TOrd>& a)
	{
		return a.begin();
	}

	template<typename T, typename TOrd>
	typename aview2d<T, TOrd>::iterator end(aview2d<T, TOrd>& a)
	{
		return a.end();
	}

	/******************************************************
	 *
	 *  View operations
	 *
	 ******************************************************/

	// element-wise comparison

	template<typename T, typename TOrd>
	inline bool operator == (const caview2d<T, TOrd>& lhs, const caview2d<T, TOrd>& rhs)
	{
		if (lhs.dim0() == rhs.dim0() && lhs.dim1() == rhs.dim1())
		{
			slice2d_info sl = lhs.slice_info();
			slice2d_info sr = rhs.slice_info();

			return _detail::elements_equal_2d(sl.nslices, sl.len, lhs.pbase(), sl.stride, rhs.pbase(), sr.stride);
		}
		else
		{
			return false;
		}
	}

	template<typename T, typename TOrd>
	inline bool operator != (const caview2d<T, TOrd>& lhs, const caview2d<T, TOrd>& rhs)
	{
		return !(lhs == rhs);
	}

	// export & import

	template<typename T, typename TOrd>
	inline void import_from(aview2d<T, TOrd>& a, const T *in)
	{
		slice2d_info s = a.slice_info();
		_detail::copy_elements_2d(s.nslices, s.len, in, (index_t)(s.len), a.pbase(), s.stride);
	}

	template<typename T, typename TOrd, typename ForwardIterator>
	inline void import_from(aview2d<T, TOrd>& a, ForwardIterator in)
	{
		std::copy_n(in, a.nelems(), a.begin());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename ForwardIterator>
	inline void import_from(aview2d_ex<T, TOrd, TIndexer0, TIndexer1>& a, ForwardIterator in)
	{
		index_t d0 = a.dim0();
		index_t d1 = a.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					a(i, j) = *in;
					++ in;
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					a(i, j) = *in;
					++ in;
				}
			}
		}
	}

	template<typename T, typename TOrd>
	inline void export_to(const caview2d<T, TOrd>& a, T *out)
	{
		slice2d_info s = a.slice_info();
		_detail::copy_elements_2d(s.nslices, s.len, a.pbase(), s.stride, out, (index_t)(s.len));
	}

	template<typename T, typename TOrd, typename ForwardIterator>
	inline void export_to(const caview2d<T, TOrd>& a, ForwardIterator out)
	{
		std::copy_n(a.begin(), a.nelems(), out);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename ForwardIterator>
	inline void export_to(const caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& a, ForwardIterator out)
	{
		index_t d0 = a.dim0();
		index_t d1 = a.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					*out = a(i, j);
					++ out;
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					*out = a(i, j);
					++ out;
				}
			}
		}
	}


	// fill

	template<typename T, typename TOrd>
	inline void set_zeros(aview2d<T, TOrd>& dst)
	{
		slice2d_info s = dst.slice_info();
		_detail::set_zeros_2d(s.nslices, s.len, dst.pbase(), s.stride);
	}

	template<typename T, typename TOrd>
	inline void fill(aview2d<T, TOrd>& dst, const T& v)
	{
		slice2d_info s = dst.slice_info();
		_detail::set_elements_2d(s.nslices, s.len, v, dst.pbase(), s.stride);
	}

	template<typename T, typename TOrd, typename TIndexer0, typename TIndexer1>
	inline void fill(aview2d_ex<T, TOrd, TIndexer0, TIndexer1>& dst, const T& v)
	{
		index_t d0 = dst.dim0();
		index_t d1 = dst.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					dst(i, j) = v;
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					dst(i, j) = v;
				}
			}
		}
	}


	// copy

	template<typename T, typename TOrd>
	inline void copy(const caview2d<T, TOrd>& src, aview2d<T, TOrd>& dst)
	{
		check_arg(src.shape() == dst.shape(), "aview2d copy: the shapes of src and dst are inconsistent.");

		slice2d_info ss = src.slice();
		slice2d_info sd = dst.slice();

		_detail::copy_elements_2d(ss.nslices, ss.len, src.pbase(), ss.stride, dst.pbase(), sd.stride);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void copy(const caview2d<T, TOrd>& src, aview2d_ex<T, TOrd, TIndexer0, TIndexer1>& dst)
	{
		check_arg(src.shape() == dst.shape(), "aview2d copy: the shapes of src and dst are inconsistent.");
		import_from(dst, src.begin());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline void copy(const caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& src, aview2d<T, TOrd>& dst)
	{
		check_arg(src.shape() == dst.shape(), "aview2d copy: the shapes of src and dst are inconsistent.");
		export_to(src, dst.begin());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline void copy(const caview2d_ex<T, TOrd, LIndexer0, LIndexer1>& src,
			aview2d_ex<T, TOrd, RIndexer0, RIndexer1>& dst)
	{
		check_arg(src.shape() == dst.shape(), "aview2d copy: the shapes of src and dst are inconsistent.");

		index_t d0 = src.dim0();
		index_t d1 = src.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					dst(i, j) = src(i, j);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					dst(i, j) = src(i, j);
				}
			}
		}
	}


	/******************************************************
	 *
	 *  stand-alone array class
	 *
	 ******************************************************/

	template<typename T, typename TOrd, class Alloc>
	class array2d : private sharable_storage_base<T, Alloc>, public aview2d<T, TOrd>
	{
	public:
		BCS_ARRAY_CHECK_TYPE(T)
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_ARRAY_BASIC_TYPEDEFS(2u, T, TOrd)

		typedef caview2d<value_type, layout_order> cview_type;
		typedef aview2d<value_type, layout_order> view_type;

		typedef sharable_storage_base<T, Alloc> storage_base_type;

	public:
		explicit array2d(size_type m, size_type n)
		: storage_base_type(m * n)
		, view_type(storage_base_type::pointer_to_base(), m, n)
		{
		}

		explicit array2d(const shape_type& shape)
		: storage_base_type(static_cast<size_type>(shape[0] * shape[1]))
		, view_type(storage_base_type::pointer_to_base(), (size_t)shape[0], (size_t)shape[1])
		{
		}

		array2d(size_type m, size_type n, const T& x)
		: storage_base_type(m * n, x)
		, view_type(storage_base_type::pointer_to_base(), m, n)
		{
		}

		array2d(size_type m, size_type n, const_pointer src)
		: storage_base_type(m * n, src)
		, view_type(storage_base_type::pointer_to_base(), m, n)
		{
		}

		array2d(const array2d& r)
		: storage_base_type(r)
		, view_type(storage_base_type::pointer_to_base(), r.nrows(), r.ncolumns())
		{
		}

		array2d(array2d&& r)
		: storage_base_type(std::move(r))
		, view_type(std::move(r))
		{
		}

		explicit array2d(const cview_type& r)
		: storage_base_type(r.nelems(), r.pbase())
		, view_type(storage_base_type::pointer_to_base(), r.nrows(), r.ncolumns())
		{
		}

		template<typename TIndexer0, typename TIndexer1>
		explicit array2d(const caview2d_ex<value_type, layout_order, TIndexer0, TIndexer1>& r)
		: storage_base_type(r.nelems())
		, view_type(storage_base_type::pointer_to_base(), r.nrows(), r.ncolumns())
		{
			copy(r, *this);
		}

		array2d& operator = (const array2d& r)
		{
			if (this != &r)
			{
				storage_base_type &s = *this;
				view_type& v = *this;

				s = r;
				v = view_type(s.pointer_to_base(), r.nrows(), r.ncolumns(), r.base_dim0(), r.base_dim1());
			}
			return *this;
		}

		array2d& operator = (array2d&& r)
		{
			storage_base_type &s = *this;
			view_type& v = *this;

			s = std::move(r);
			v = std::move(r);

			return *this;
		}

		void swap(array2d& r)
		{
			using std::swap;

			storage_base_type::swap(r);

			view_type& v = *this;
			view_type& rv = r;
			swap(v, rv);
		}

		// sharing

	public:
		array2d(const array2d& r, do_share ds)
		: storage_base_type(r, ds)
		, view_type(storage_base_type::pointer_to_base(), r.nrows(), r.ncolumns(), r.base_dim0(), r.base_dim1())
		{
		}

		array2d shared_copy() const
		{
			return array2d(*this, do_share());
		}

	}; // end class array2d

	template<typename T, typename TOrd, class Alloc>
	inline void swap(array2d<T, TOrd, Alloc>& lhs, array2d<T, TOrd, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}

	template<typename T, typename TOrd>
	inline array2d<T, TOrd> clone_array(const caview2d<T, TOrd>& a)
	{
		return array2d<T, TOrd>(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> clone_array(const caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return array2d<T, TOrd>(a);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	template<typename T, typename TOrd, class IndexIter>
	array1d<T> select_elems(const caview2d<T, TOrd>& a, size_t n, IndexIter I, IndexIter J)
	{
		array1d<T> r(n);

		index_t d = r.dim0();
		for (index_t i = 0; i < d; ++i, ++I, ++J)
		{
			r(i) = a(*I, *J);
		}

		return r;
	}


	template<typename T, typename TOrd, class IndexIter>
	array2d<T, TOrd> select_rows(const caview2d<T, TOrd>& a, size_t m, IndexIter I)
	{
		array2d<T, TOrd> r(m, a.ncolumns());

		index_t d0 = r.dim0();
		index_t d1 = r.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i, ++I)
			{
				index_t si = *I;
				for (index_t j = 0; j < d1; ++j)
				{
					r(i, j) = a(si, j);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				IndexIter _I(I);
				for (index_t i = 0; i < d0; ++i, ++_I)
				{
					r(i, j) = a(*_I, j);
				}
			}
		}

		return r;
	}


	template<typename T, typename TOrd, class IndexIter>
	array2d<T, TOrd> select_columns(const caview2d<T, TOrd>& a, size_t n, IndexIter J)
	{
		array2d<T, TOrd> r(a.nrows(), n);

		index_t d0 = r.dim0();
		index_t d1 = r.dim1();

		if (std::is_same<TOrd, column_major_t>::value)
		{
			for (index_t j = 0; j < d1; ++j, ++J)
			{
				index_t sj = *J;
				for (index_t i = 0; i < d0; ++i)
				{
					r(i, j) = a(i, sj);
				}
			}
		}
		else
		{
			for (index_t i = 0; i < d0; ++i)
			{
				IndexIter _J(J);
				for (index_t j = 0; j < d1; ++j, ++_J)
				{
					r(i, j) = a(i, *_J);
				}
			}
		}

		return r;
	}


	template<typename T, typename TOrd, class IndexIter0, class IndexIter1>
	array2d<T, TOrd> select_rows_and_cols(const caview2d<T, TOrd>& a, size_t m, IndexIter0 I, size_t n, IndexIter1 J)
	{
		array2d<T, TOrd> r(m, n);

		index_t d0 = r.dim0();
		index_t d1 = r.dim1();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i, ++I)
			{
				index_t si = *I;
				IndexIter1 _J(J);

				for (index_t j = 0; j < d1; ++j, ++_J)
				{
					r(i, j) = a(si, *_J);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j, ++J)
			{
				index_t sj = *J;
				IndexIter0 _I(I);

				for (index_t i = 0; i < d0; ++i, ++_I)
				{
					r(i, j) = a(*_I, sj);
				}
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
	inline void transpose_matrix(const T *src, T *dst, size_t m, size_t n)
	{
		const size_t block_size = BCS_TRANSPOSITION_BLOCK_BYTES / sizeof(T);

		aligned_array<T, block_size> cache;

		if (block_size < 4)
		{
			direct_transpose_matrix(src, dst, m, n);
		}
		else
		{
			size_t bdim = (size_t)std::sqrt((double)block_size);
			blockwise_transpose_matrix(src, dst, m, n, bdim, cache.data);
		}
	}

	template<typename T, typename TOrd>
	inline array2d<T, TOrd> transpose(const caview2d<T, TOrd>& a)
	{
		array2d<T, TOrd> r(a.ncolumns(), a.nrows());

		slice2d_info sli = a.slice_info();
		transpose_matrix(a.pbase(), r.pbase(), sli.nslices, sli.len);

		return r;
	}

}


#endif



