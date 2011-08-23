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

#define BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd) \
	typedef typename bcs::_detail::slice_helper2d<T, TOrd>::row_cview_type row_cview_type; \
	typedef typename bcs::_detail::slice_helper2d<T, TOrd>::row_view_type row_view_type; \
	typedef typename bcs::_detail::slice_helper2d<T, TOrd>::column_cview_type column_cview_type; \
	typedef typename bcs::_detail::slice_helper2d<T, TOrd>::column_view_type column_view_type;


namespace bcs
{
	/**************************************************
	 *
	 *  layout order specific indexing implementation
	 *
	 **************************************************/

	inline BCS_ENSURE_INLINE index_t sub2ind(const extent2d_t& extent, index_t i, index_t j, row_major_t)
	{
		return i * extent[1] + j;
	}

	inline BCS_ENSURE_INLINE index_t sub2ind(const extent2d_t& extent, index_t i, index_t j, column_major_t)
	{
		return i + extent[0] * j;
	}

	/******************************************************
	 *
	 *  Slice-views
	 *
	 ******************************************************/

	namespace _detail
	{
		template<typename T, typename TOrd> struct slice_helper2d;

		template<typename T>
		struct slice_helper2d<T, row_major_t>
		{
			typedef caview1d<T> row_cview_type;
			typedef aview1d<T>  row_view_type;
			typedef caview1d_ex<T, step_ind> column_cview_type;
			typedef aview1d_ex<T, step_ind>  column_view_type;

			static row_cview_type row_cview(const T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow)
			{
				return caview1d<T>(pbase + irow * ext[1], d1);
			}

			static row_view_type row_view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow)
			{
				return aview1d<T>(pbase + irow * ext[1], d1);
			}

			static column_cview_type column_cview(const T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol)
			{
				return caview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, ext[1]));
			}

			static column_view_type column_view(T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol)
			{
				return aview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, ext[1]));
			}
		};


		template<typename T>
		struct slice_helper2d<T, column_major_t>
		{
			typedef caview1d<T> column_cview_type;
			typedef aview1d<T>  column_view_type;
			typedef caview1d_ex<T, step_ind> row_cview_type;
			typedef aview1d_ex<T, step_ind>  row_view_type;

			static row_cview_type row_cview(const T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow)
			{
				return caview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, ext[0]));
			}

			static row_view_type row_view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow)
			{
				return aview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, ext[0]));
			}

			static column_cview_type column_cview(const T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol)
			{
				return caview1d<T>(pbase + icol * ext[0], d0);
			}

			static column_view_type column_view(T* pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol)
			{
				return aview1d<T>(pbase + icol * ext[0], d0);
			}
		};


		template<typename T, typename TOrd, class TRange> struct slice_row_range_helper2d;
		template<typename T, typename TOrd, class TRange> struct slice_col_range_helper2d;

		template<typename T, class TRange>
		struct slice_row_range_helper2d<T, row_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  view_type;

			static cview_type cview(const T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return cview_type(pbase + ext[1] * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}

			static view_type view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return view_type(pbase + ext[1] * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}
		};

		template<typename T, class TRange>
		struct slice_col_range_helper2d<T, row_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  view_type;

			static cview_type cview(const T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return cview_type(pbase + ext[1] * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, ext[1], rgn));
			}

			static view_type view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return view_type(pbase + ext[1] * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, ext[1], rgn));
			}
		};


		template<typename T, class TRange>
		struct slice_row_range_helper2d<T, column_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  view_type;

			static cview_type cview(const T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return cview_type(pbase + ext[0] * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, ext[0], rgn));
			}

			static view_type view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return view_type(pbase + ext[0] * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, ext[0], rgn));
			}
		};

		template<typename T, class TRange>
		struct slice_col_range_helper2d<T, column_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  view_type;

			static cview_type cview(const T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return cview_type(pbase + ext[0] * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}

			static view_type view(T *pbase, const extent2d_t& ext, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return view_type(pbase + ext[0] * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}
		};
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::row_cview_type
	row_view(const dense_caview2d_base<Derived>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::row_cview(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::row_view_type
	row_view(dense_aview2d_base<Derived>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::row_view(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::column_cview_type
	column_view(const dense_caview2d_base<Derived>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_cview(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), icol);
	}

	template<class Derived>
	inline typename _detail::slice_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order>::column_view_type
	column_view(dense_aview2d_base<Derived>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_view(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), icol);
	}


	template<class Derived, class TRange>
	inline typename _detail::slice_row_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::cview_type
	row_view(const dense_caview2d_base<Derived>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<typename Derived::value_type,
				typename Derived::layout_order, TRange> helper;

		return helper::cview(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_row_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::view_type
	row_view(dense_aview2d_base<Derived>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<typename Derived::value_type,
				typename Derived::layout_order, TRange> helper;

		return helper::view(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_col_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::cview_type
	column_view(const dense_caview2d_base<Derived>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<typename Derived::value_type,
				typename Derived::layout_order, TRange> helper;

		return helper::cview(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), icol, rgn);
	}

	template<class Derived, class TRange>
	inline typename _detail::slice_col_range_helper2d<
		typename Derived::value_type,
		typename Derived::layout_order,
		TRange>::view_type
	column_view(dense_aview2d_base<Derived>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<typename Derived::value_type,
				typename Derived::layout_order, TRange> helper;

		return helper::view(a.pbase(), a.base_extent(), a.dim0(), a.dim1(), icol, rgn);
	}


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
		caview2d_ex(const_pointer pbase, const extent2d_t& base_ext, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(pbase)
		, m_base_ext(base_ext)
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

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0, m_d1);
		}

		BCS_ENSURE_INLINE extent2d_t base_extent() const
		{
			return m_base_ext;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

	private:
		index_type offset(index_type i, index_type j) const
		{
			return sub2ind(m_base_ext, m_indexer0[i], m_indexer1[j], TOrd());
		}

		const_pointer m_pbase;
		extent2d_t m_base_ext;
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
		aview2d_ex(pointer pbase, const extent2d_t& base_ext, const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(pbase)
		, m_base_ext(base_ext)
		, m_d0(indexer0.dim())
		, m_d1(indexer1.dim())
		, m_indexer0(indexer0)
		, m_indexer1(indexer1)
		{
		}

		operator caview2d_ex<T, TOrd, TIndexer0, TIndexer1>() const
		{
			return caview2d_ex<T, TOrd, TIndexer0, TIndexer1>(
					m_pbase, m_base_ext, m_indexer0, m_indexer1);
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

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_d0, m_d1);
		}

		BCS_ENSURE_INLINE shape_type base_extent() const
		{
			return m_base_ext;
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
			return sub2ind(m_base_ext, m_indexer0[i], m_indexer1[j], TOrd());
		}

		pointer m_pbase;
		extent2d_t m_base_ext;
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

		return ret_type(&(a(i0, j0)), make_extent2d(a.dim0(), a.dim1()),
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

		return ret_type(&(a(i0, j0)), make_extent2d(a.dim0(), a.dim1()),
				indexer_map<IndexSelector0>::get_indexer(a.dim0(), I),
				indexer_map<IndexSelector1>::get_indexer(a.dim1(), J));
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

		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class caview2d : public continuous_caview2d_base<caview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		caview2d(const_pointer pbase, index_type m, index_type n)
		: m_pbase(pbase), m_ext(make_extent2d(m, n))
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape)
		: m_pbase(pbase), m_ext(shape)
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
			return dim0() * dim1();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return dim0() == 0 && dim1() == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_ext[0];
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_ext[1];
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
			return arr_shape(dim0(), dim1());
		}

		BCS_ENSURE_INLINE extent2d_t base_extent() const
		{
			return m_ext;
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
			return m_pbase[sub2ind(m_ext, i, j, TOrd())];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_view(*this, i, rgn);
		}

		column_cview_type column(index_t i) const
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
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
		extent2d_t m_ext;

	}; // end class caview2d


	template<typename T, typename TOrd>
	struct aview_traits<aview2d<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		static const bool is_dense = true;
		static const bool is_continuous = true;
		static const bool is_const_view = false;

		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class aview2d : public continuous_aview2d_base<aview2d<T, TOrd> >
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		aview2d(pointer pbase, index_type m, index_type n)
		: m_pbase(pbase), m_ext(make_extent2d(m, n))
		{
		}

		aview2d(pointer pbase, const shape_type& shape)
		: m_pbase(pbase), m_ext(shape)
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
			return dim0() * dim1();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return dim0() == 0 && dim1() == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return m_ext[0];
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_ext[1];
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
			return arr_shape(dim0(), dim1());
		}

		BCS_ENSURE_INLINE extent2d_t base_extent() const
		{
			return m_ext;
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
			return m_pbase[sub2ind(m_ext, i, j, TOrd())];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return m_pbase[sub2ind(m_ext, i, j, TOrd())];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_view(*this, i);
		}

		row_view_type row(index_t i)
		{
			return row_view(*this, i);
		}

		column_cview_type column(index_t i) const
		{
			return column_view(*this, i);
		}

		column_view_type column(index_t i)
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::view_type
		row(index_t i, const TRange& rgn)
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
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
		extent2d_t m_ext;

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
