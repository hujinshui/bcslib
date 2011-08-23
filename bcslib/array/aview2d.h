/**
 * @file aview2d.h
 *
 * The classes to represent two-dimensional views
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_AVIEW2D_H_
#define BCSLIB_AVIEW2D_H_

#include <bcslib/array/aview2d_base.h>
#include <bcslib/array/aview1d.h>

namespace bcs
{

	/******************************************************
	 *
	 *  Extended Views
	 *
	 ******************************************************/

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview_traits<caview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		static const bool is_dense = true;
		static const bool is_continuous = false;
		static const bool is_const_view = true;
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class caview2d_ex
	: public dense_caview2d_base<caview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );

		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	public:
		caview2d_ex(const_pointer pbase, index_t base_d0, index_t base_d1, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(pbase)
		, m_idxcore(base_d0, base_d1)
		, m_d0(indexer0.dim())
		, m_d1(indexer1.dim())
		, m_indexer0(indexer0)
		, m_indexer1(indexer1)
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
			return m_d0 * m_d1;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0 || m_d1 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_d1;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_d1;
		}

		BCS_ENSURE_INLINE index_type base_dim0() const
		{
			return m_idxcore.base_dim0();
		}

		BCS_ENSURE_INLINE index_type base_dim1() const
		{
			return m_idxcore.base_dim1();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0, m_d1);
		}

		BCS_ENSURE_INLINE shape_type base_shape() const
		{
			return m_idxcore.base_shape();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

	private:
		index_type offset(index_type i, index_type j) const
		{
			return m_idxcore.offset(m_indexer0[i], m_indexer1[j]);
		}

		const_pointer m_pbase;
		_detail::index_core2d<layout_order> m_idxcore;
		index_t m_d0;
		index_t m_d1;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class caview2d_ex



	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview_traits<aview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		static const bool is_dense = true;
		static const bool is_continuous = false;
		static const bool is_const_view = false;
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d_ex
	: public dense_aview2d_base<aview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );

		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	public:
		aview2d_ex(pointer pbase, index_t base_d0, index_t base_d1, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(pbase)
		, m_idxcore(base_d0, base_d1)
		, m_d0(indexer0.dim())
		, m_d1(indexer1.dim())
		, m_indexer0(indexer0)
		, m_indexer1(indexer1)
		{
		}

		operator caview2d_ex<T, TOrd, TIndexer0, TIndexer1>() const
		{
			return caview2d_ex<T, TOrd, TIndexer0, TIndexer1>(
					m_pbase, base_dim0(), base_dim1(), m_indexer0, m_indexer1);
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
			return m_d0 * m_d1;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_d0 == 0 || m_d1 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_d1;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_d1;
		}

		BCS_ENSURE_INLINE index_type base_dim0() const
		{
			return m_idxcore.base_dim0();
		}

		BCS_ENSURE_INLINE index_type base_dim1() const
		{
			return m_idxcore.base_dim1();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0, m_d1);
		}

		BCS_ENSURE_INLINE shape_type base_shape() const
		{
			return m_idxcore.base_shape();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return m_pbase[offset(i, j)];
		}

	private:
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

	}; // end class aview2d_ex


	/******************************************************
	 *
	 *  Sub-views
	 *
	 ******************************************************/

	template<class Derived, class IndexSelector0, class IndexSelector1>
	inline caview2d_ex<
		typename Derived::value_type,
		typename Derived::layout_order,
		typename indexer_map<IndexSelector0>::type,
		typename indexer_map<IndexSelector1>::type>
	subview(const dense_caview2d_base<Derived>& a, const IndexSelector0& I, const IndexSelector1& J)
	{
		typedef caview2d_ex<
			typename Derived::value_type,
			typename Derived::layout_order,
			typename indexer_map<IndexSelector0>::type,
			typename indexer_map<IndexSelector1>::type> ret_type;

		index_t i0 = indexer_map<IndexSelector0>::get_offset(a.dim0(), I);
		index_t j0 = indexer_map<IndexSelector1>::get_offset(a.dim1(), J);

		return ret_type(&(a(i0, j0)), a.dim0(), a.dim1(),
				indexer_map<IndexSelector0>::get_indexer(a.dim0(), I),
				indexer_map<IndexSelector1>::get_indexer(a.dim1(), J));
	}

	template<class Derived, class IndexSelector0, class IndexSelector1>
	inline aview2d_ex<
		typename Derived::value_type,
		typename Derived::layout_order,
		typename indexer_map<IndexSelector0>::type,
		typename indexer_map<IndexSelector1>::type>
	subview(dense_aview2d_base<Derived>& a, const IndexSelector0& I, const IndexSelector1& J)
	{
		typedef aview2d_ex<
			typename Derived::value_type,
			typename Derived::layout_order,
			typename indexer_map<IndexSelector0>::type,
			typename indexer_map<IndexSelector1>::type> ret_type;

		index_t i0 = indexer_map<IndexSelector0>::get_offset(a.dim0(), I);
		index_t j0 = indexer_map<IndexSelector1>::get_offset(a.dim1(), J);

		return ret_type(&(a(i0, j0)), a.dim0(), a.dim1(),
				indexer_map<IndexSelector0>::get_indexer(a.dim0(), I),
				indexer_map<IndexSelector1>::get_indexer(a.dim1(), J));
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::row_cview_type
	row_view(const dense_caview2d_base<Derived>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::row_cview(a.pbase(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::row_view_type
	row_view(dense_aview2d_base<Derived>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::row_view(a.pbase(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::column_cview_type
	column_view(const dense_caview2d_base<Derived>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_cview(a.pbase(), a.dim0(), a.dim1(), icol);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::column_view_type
	column_view(dense_aview2d_base<Derived>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_view(a.pbase(), a.dim0(), a.dim1(), icol);
	}


	template<class Derived, class TRange>
	inline typename _detail::slice_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::row_range_cview_type
	row_view(const dense_caview2d_base<Derived>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_range_helper2d<
				typename Derived::value_type,
				typename Derived::layout_order,
				TRange> helper;

		return helper::row_range_cview(a.pbase(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::row_range_view_type
	row_view(dense_aview2d_base<Derived>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_range_helper2d<
				typename Derived::value_type,
				typename Derived::layout_order,
				TRange> helper;

		return helper::row_range_view(a.pbase(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::column_range_cview_type
	column_view(const dense_caview2d_base<Derived>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_range_helper2d<
				typename Derived::value_type,
				typename Derived::layout_order,
				TRange> helper;

		return helper::column_range_cview(a.pbase(), a.dim0(), a.dim1(), icol, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::column_range_view_type
	column_view(dense_aview2d_base<Derived>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_range_helper2d<
				typename Derived::value_type,
				typename Derived::layout_order,
				TRange> helper;

		return helper::column_range_view(a.pbase(), a.dim0(), a.dim1(), icol, rgn);
	}


	/******************************************************
	 *
	 *  Dense views
	 *
	 ******************************************************/

	template<typename T, typename TOrd>
	struct aview_traits<caview2d<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		static const bool is_dense = true;
		static const bool is_continuous = true;
		static const bool is_const_view = true;
	};

	template<typename T, typename TOrd>
	class caview2d : public continuous_caview2d_base<caview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

	public:
		caview2d(const_pointer pbase, index_type m, index_type n)
		: m_pbase(pbase), m_idxcore(m, n)
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape)
		: m_pbase(pbase), m_idxcore(shape[0], shape[1])
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
			return m_idxcore.base_nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return dim0() == 0 && dim1() == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_idxcore.base_dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_idxcore.base_dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return dim0();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return dim1();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return m_idxcore.base_shape();
		}

	public:
		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return m_pbase[i];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

	public:
		typename _detail::slice_helper2d<value_type, layout_order>::row_cview_type
		row(index_t i) const
		{
			return row_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::row_range_cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_view(*this, i, rgn);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_cview_type
		column(index_t i) const
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::column_range_cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		caview2d_ex<value_type, layout_order,
			typename indexer_map<TRange0>::type,
			typename indexer_map<TRange1>::type>
		V(const TRange0& I, const TRange1& J) const
		{
			return subview(*this, I, J);
		}

	private:
		const_pointer m_pbase;
		_detail::index_core2d<layout_order> m_idxcore;

	}; // end class caview2d


	template<typename T, typename TOrd>
	struct aview_traits<aview2d<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		static const bool is_dense = true;
		static const bool is_continuous = true;
		static const bool is_const_view = false;
	};

	template<typename T, typename TOrd>
	class aview2d : public continuous_aview2d_base<aview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

	public:
		aview2d(pointer pbase, index_type m, index_type n)
		: m_pbase(pbase), m_idxcore(m, n)
		{
		}

		aview2d(pointer pbase, const shape_type& shape)
		: m_pbase(pbase), m_idxcore(shape[0], shape[1])
		{
		}

		operator caview2d<T, TOrd>() const
		{
			return caview2d<T, TOrd>(m_pbase, dim0(), dim1());
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
			return m_idxcore.base_nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return dim0() == 0 && dim1() == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_idxcore.base_dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_idxcore.base_dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return dim0();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return dim1();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return m_idxcore.base_shape();
		}

	public:
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return m_pbase[m_idxcore.offset(i, j)];
		}

	public:
		typename _detail::slice_helper2d<value_type, layout_order>::row_cview_type
		row(index_t i) const
		{
			return row_view(*this, i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::row_view_type
		row(index_t i)
		{
			return row_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::row_range_cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::row_range_view_type
		row(index_t i, const TRange& rgn)
		{
			return row_view(*this, i, rgn);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_cview_type
		column(index_t i) const
		{
			return column_view(*this, i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_view_type
		column(index_t i)
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::column_range_cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::column_range_view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		caview2d_ex<value_type, layout_order,
			typename indexer_map<TRange0>::type,
			typename indexer_map<TRange1>::type>
		V(const TRange0& I, const TRange1& J) const
		{
			return subview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		aview2d_ex<value_type, layout_order,
			typename indexer_map<TRange0>::type,
			typename indexer_map<TRange1>::type>
		V(const TRange0& I, const TRange1& J)
		{
			return subview(*this, I, J);
		}

	private:
		pointer m_pbase;
		_detail::index_core2d<layout_order> m_idxcore;

	}; // end class aview2d


	// convenient functions

	template<typename T>
	inline caview2d<T, row_major_t> make_caview2d_rm(const T *base, index_t m, index_t n)
	{
		return caview2d<T, row_major_t>(base, m, n);
	}

	template<typename T>
	inline aview2d<T, row_major_t> make_aview2d_rm(T *base, index_t m, index_t n)
	{
		return aview2d<T, row_major_t>(base, m, n);
	}

	template<typename T>
	inline caview2d<T, column_major_t> make_caview2d_cm(const T *base, index_t m, index_t n)
	{
		return caview2d<T, column_major_t>(base, m, n);
	}

	template<typename T>
	inline aview2d<T, column_major_t> make_aview2d_cm(T *base, index_t m, index_t n)
	{
		return aview2d<T, column_major_t>(base, m, n);
	}




}

#endif
