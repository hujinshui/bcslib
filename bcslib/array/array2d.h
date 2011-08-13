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

#include <bcslib/array/array2d_base.h>
#include <bcslib/array/array1d.h>

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

		typedef caview2d_ex<T, TOrd, TIndexer0, TIndexer1> self_type;
		typedef caview2d_base<self_type> view_nd_base;
		typedef caview_base<self_type> dview_base;
		typedef caview_base<self_type> view_base;
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class caview2d_ex
	: public caview2d_base<caview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

	public:
		void export_to(pointer dst) const
		{
			index_t d0 = dim0();
			index_t d1 = dim1();

			if (std::is_same<layout_order, row_major_t>::value)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					for (index_t j = 0; j < d1; ++j)
					{
						*(dst++) = operator()(i, j);
					}
				}
			}
			else
			{
				for (index_t j = 0; j < d1; ++j)
				{
					for (index_t i = 0; i < d0; ++i)
					{
						*(dst++) = operator()(i, j);
					}
				}
			}
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

		typedef aview2d_ex<T, TOrd, TIndexer0, TIndexer1> self_type;
		typedef aview2d_base<self_type> view_nd_base;
		typedef aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d_ex
	: public aview2d_base<aview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

	public:
		void export_to(pointer dst) const
		{
			index_t d0 = dim0();
			index_t d1 = dim1();

			if (std::is_same<layout_order, row_major_t>::value)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					for (index_t j = 0; j < d1; ++j)
					{
						*(dst++) = operator()(i, j);
					}
				}
			}
			else
			{
				for (index_t j = 0; j < d1; ++j)
				{
					for (index_t i = 0; i < d0; ++i)
					{
						*(dst++) = operator()(i, j);
					}
				}
			}
		}

		void import_from(const_pointer src)
		{
			index_t d0 = dim0();
			index_t d1 = dim1();

			if (std::is_same<layout_order, row_major_t>::value)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					for (index_t j = 0; j < d1; ++j)
					{
						operator()(i, j) = *(src++);
					}
				}
			}
			else
			{
				for (index_t j = 0; j < d1; ++j)
				{
					for (index_t i = 0; i < d0; ++i)
					{
						operator()(i, j) = *(src++);
					}
				}
			}
		}

		void fill(const value_type& v)
		{
			index_t d0 = dim0();
			index_t d1 = dim1();

			if (std::is_same<layout_order, row_major_t>::value)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					for (index_t j = 0; j < d1; ++j)
					{
						operator()(i, j) = v;
					}
				}
			}
			else
			{
				for (index_t j = 0; j < d1; ++j)
				{
					for (index_t i = 0; i < d0; ++i)
					{
						operator()(i, j) = v;
					}
				}
			}
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

		typedef caview2d<T, TOrd> self_type;
		typedef caview2d_base<self_type> view_nd_base;
		typedef dense_caview_base<self_type> dview_base;
		typedef caview_base<self_type> view_base;
	};

	template<typename T, typename TOrd>
	class caview2d : public dense_caview2d_base<caview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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

		void export_to(pointer dst) const
		{
			copy_elements(pbase(), dst, size());
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

		typedef aview2d<T, TOrd> self_type;
		typedef aview2d_base<self_type> view_nd_base;
		typedef dense_aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};

	template<typename T, typename TOrd>
	class aview2d : public dense_aview2d_base<aview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
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


	// convenient functions to make 2D view

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


	/******************************************************
	 *
	 *  View operations
	 *
	 ******************************************************/

	// comparison

	template<class LDerived, class RDerived>
	inline bool is_same_shape(const caview2d_base<LDerived>& lhs, const caview2d_base<RDerived>& rhs)
	{
		return lhs.dim0() == rhs.dim0() && lhs.dim1() == rhs.dim1();
	}

	template<class LDerived, class RDerived>
	inline bool is_equal(const dense_caview2d_base<LDerived>& lhs, const dense_caview2d_base<RDerived>& rhs)
	{
		return is_same_shape(lhs, rhs) && elements_equal(lhs.pbase(), rhs.pbase(), lhs.size());
	}

	// copy

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview2d_base<LDerived>& src, dense_aview2d_base<RDerived>& dst)
	{
		BCS_CHECK_LAYOUT_ORD2(typename LDerived, typename RDerived)

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.size());
	}

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst)
	{
		BCS_CHECK_LAYOUT_ORD2(typename LDerived, typename RDerived)

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		dst.import_from(src.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview2d_base<LDerived>& src, dense_aview2d_base<RDerived>& dst)
	{
		BCS_CHECK_LAYOUT_ORD2(typename LDerived, typename RDerived)

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		src.export_to(dst.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst)
	{
		BCS_CHECK_LAYOUT_ORD2(typename LDerived, typename RDerived)

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");

		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		index_t d0 = src.dim0();
		index_t d1 = src.dim1();

		if (std::is_same<typename LDerived::layout_order, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					dstd(i, j) = srcd(i, j);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					dstd(i, j) = srcd(i, j);
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
	struct aview_traits<array2d<T, TOrd, Alloc> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef array2d<T, TOrd, Alloc> self_type;
		typedef aview2d_base<self_type> view_nd_base;
		typedef dense_aview_base<self_type> dview_base;
		typedef aview_base<self_type> view_base;
	};


	template<typename T, typename TOrd, class Alloc=aligned_allocator<T> > class array2d;

	template<typename T, typename TOrd, class Alloc>
	class array2d
	: private sharable_storage_base<T, Alloc>
	, public dense_aview2d_base<array2d<T, TOrd, Alloc> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_valid_array_value<T> );
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef sharable_storage_base<T, Alloc> storage_base;
		typedef aview2d<T, TOrd> view_type;

	public:
		explicit array2d(index_type m, index_type n)
		: storage_base((size_t)(m * n))
		, m_view(storage_base::pointer_to_base(), m, n)
		{
		}

		explicit array2d(const shape_type& shape)
		: storage_base((size_t)(shape[0] * shape[1]))
		, m_view(storage_base::pointer_to_base(), shape[0], shape[1])
		{
		}

		array2d(index_type m, index_type n, const T& x)
		: storage_base((size_t)(m * n), x)
		, m_view(storage_base::pointer_to_base(), m, n)
		{
		}

		array2d(index_type m, index_type n, const_pointer src)
		: storage_base((size_t)(m * n), src)
		, m_view(storage_base::pointer_to_base(), m, n)
		{
		}

		array2d(const array2d& r)
		: storage_base(r)
		, m_view(storage_base::pointer_to_base(), r.nrows(), r.ncolumns())
		{
		}

		array2d(array2d&& r)
		: storage_base(std::move(r))
		, m_view(std::move(r.m_view))
		{
			r.m_view = view_type(BCS_NULL, 0, 0);
		}

		template<class Derived>
		explicit array2d(const caview2d_base<Derived>& r)
		: storage_base(r.size())
		, m_view(storage_base::pointer_to_base(), r.nrows(), r.ncolumns())
		{
			copy(r.derived(), *this);
		}


		array2d(const array2d& r, do_share ds)
		: storage_base(r, ds)
		, m_view(storage_base::pointer_to_base(), r.nrows(), r.ncolumns())
		{
		}

		array2d shared_copy() const
		{
			return array2d(*this, do_share());
		}

		array2d& operator = (const array2d& r)
		{
			if (this != &r)
			{
				storage_base &s = *this;
				s = r;
				m_view = view_type(s.pointer_to_base(), r.nrows(), r.ncolumns());
			}
			return *this;
		}

		array2d& operator = (array2d&& r)
		{
			storage_base &s = *this;
			s = std::move(r);
			m_view = std::move(r.m_view);
			r.m_view = view_type(BCS_NULL, 0, 0);

			return *this;
		}

		void swap(array2d& r)
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

			index_t m = dim0();
			index_t n = dim1();
			m_view = view_type(storage_base::pointer_to_base(), m, n);
		}

		operator caview2d<T, TOrd>() const
		{
			return caview2d<T, TOrd>(pbase(), dim0(), dim1());
		}

		operator aview2d<T, TOrd>()
		{
			return aview2d<T, TOrd>(pbase(), dim0(), dim1());
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

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_view.dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_view.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_view.ncolumns();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return m_view.shape();
		}

	public:
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_view(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return m_view(i, j);
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

	public:
		typename _detail::slice_helper2d<value_type, layout_order>::row_cview_type
		row(index_t i) const
		{
			return m_view.row(i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::row_view_type
		row(index_t i)
		{
			return m_view.row(i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::row_range_cview_type
		row(index_t i, const TRange& rgn) const
		{
			return m_view.row(i, rgn);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::row_range_view_type
		row(index_t i, const TRange& rgn)
		{
			return m_view.row(i, rgn);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_cview_type
		column(index_t i) const
		{
			return m_view.column(i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_view_type
		column(index_t i)
		{
			return m_view.column(i);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::column_range_cview_type
		column(index_t i, const TRange& rgn) const
		{
			return m_view.column(i, rgn);
		}

		template<class TRange>
		typename _detail::slice_range_helper2d<value_type, layout_order, TRange>::column_range_view_type
		column(index_t i, const TRange& rgn)
		{
			return m_view.column(i, rgn);
		}

		template<class TRange0, class TRange1>
		caview2d_ex<value_type, layout_order,
			typename indexer_map<TRange0>::type,
			typename indexer_map<TRange1>::type>
		V(const TRange0& I, const TRange1& J) const
		{
			return m_view.V(I, J);
		}

		template<class TRange0, class TRange1>
		aview2d_ex<value_type, layout_order,
			typename indexer_map<TRange0>::type,
			typename indexer_map<TRange1>::type>
		V(const TRange0& I, const TRange1& J)
		{
			return m_view.V(I, J);
		}

	private:
		view_type m_view;

	}; // end class array2d


	template<typename T, typename TOrd, class Alloc>
	inline void swap(array2d<T, TOrd, Alloc>& lhs, array2d<T, TOrd, Alloc>& rhs)
	{
		lhs.swap(rhs);
	}

	template<class Derived>
	inline array2d<typename Derived::value_type, typename Derived::layout_order>
	clone_array(const caview2d_base<Derived>& a)
	{
		return array2d<typename Derived::value_type, typename Derived::layout_order>(a);
	}


	/******************************************************
	 *
	 *  Element selection
	 *
	 ******************************************************/

	template<class Derived, class IndexSelector>
	array1d<typename Derived::value_type>
	select_elems(const caview2d_base<Derived>& a, const IndexSelector& I, const IndexSelector& J)
	{
		index_t n = (index_t)I.size();
		check_arg(n == (index_t)J.size(), "Inconsistent selector sizes.");

		array1d<typename Derived::value_type> r(n);

		const Derived& ad = a.derived();

		for (index_t i = 0; i < n; ++i)
		{
			r[i] = ad(I[i], J[i]);
		}
		return r;
	}


	template<class Derived, class IndexSelector>
	array2d<typename Derived::value_type, typename Derived::layout_order>
	select_rows(const caview2d_base<Derived>& a, const IndexSelector& I)
	{
		typedef typename Derived::value_type T;
		typedef typename Derived::layout_order TOrd;

		index_t m = (index_t)I.size();
		index_t n = a.ncolumns();
		array2d<T, TOrd> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < m; ++i)
			{
				index_t si = I[i];
				for (index_t j = 0; j < n; ++j)
				{
					*(rp++) = ad(si, j);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					*(rp++) = ad(I[i], j);
				}
			}
		}

		return r;
	}


	template<class Derived, class IndexSelector>
	array2d<typename Derived::value_type, typename Derived::layout_order>
	select_columns(const caview2d_base<Derived>& a, const IndexSelector& J)
	{
		typedef typename Derived::value_type T;
		typedef typename Derived::layout_order TOrd;

		index_t m = a.nrows();
		index_t n = (index_t)J.size();
		array2d<T, TOrd> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					*(rp++) = ad(i, J[j]);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < n; ++j)
			{
				index_t sj = J[j];
				for (index_t i = 0; i < m; ++i)
				{
					*(rp++) = ad(i, sj);
				}
			}
		}

		return r;
	}


	template<class Derived, class IndexSelectorI, class IndexSelectorJ>
	array2d<typename Derived::value_type, typename Derived::layout_order>
	select_rows_cols(const caview2d_base<Derived>& a, const IndexSelectorI& I, const IndexSelectorJ& J)
	{
		typedef typename Derived::value_type T;
		typedef typename Derived::layout_order TOrd;

		index_t m = (index_t)I.size();
		index_t n = (index_t)J.size();
		array2d<T, TOrd> r(m, n);

		const Derived& ad = a.derived();
		T* rp = r.pbase();

		if (std::is_same<TOrd, row_major_t>::value)
		{
			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					*(rp++) = ad(I[i], J[j]);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					*(rp++) = ad(I[i], J[j]);
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
	inline void transpose_matrix(const T *src, T *dst, index_t m, index_t n)
	{
		const size_t block_size = BCS_TRANSPOSITION_BLOCK_BYTES / sizeof(T);

		aligned_array<T, block_size> cache;

		if (block_size < 4)
		{
			direct_transpose_matrix(src, dst, (size_t)m, (size_t)n);
		}
		else
		{
			size_t bdim = (size_t)std::sqrt((double)block_size);
			blockwise_transpose_matrix(src, dst, (size_t)m, (size_t)n, bdim, cache.data);
		}
	}

	template<class Derived>
	inline array2d<typename Derived::value_type, typename Derived::layout_order>
	transpose(const dense_caview2d_base<Derived>& a)
	{
		array2d<typename Derived::value_type, typename Derived::layout_order> r(a.ncolumns(), a.nrows());

		if (std::is_same<typename Derived::layout_order, row_major_t>::value)
		{
			transpose_matrix(a.pbase(), r.pbase(), a.nrows(), a.ncolumns());
		}
		else
		{
			transpose_matrix(a.pbase(), r.pbase(), a.ncolumns(), a.nrows());
		}

		return r;
	}

}


#endif



