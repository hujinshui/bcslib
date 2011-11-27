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

			static row_cview_type row_cview(const T* pbase, index_t ldim, index_t d0, index_t d1, index_t irow)
			{
				return caview1d<T>(pbase + irow * ldim, d1);
			}

			static row_view_type row_view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow)
			{
				return aview1d<T>(pbase + irow * ldim, d1);
			}

			static column_cview_type column_cview(const T* pbase, index_t ldim, index_t d0, index_t d1, index_t icol)
			{
				return caview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, ldim));
			}

			static column_view_type column_view(T* pbase, index_t ldim, index_t d0, index_t d1, index_t icol)
			{
				return aview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, ldim));
			}
		};


		template<typename T>
		struct slice_helper2d<T, column_major_t>
		{
			typedef caview1d<T> column_cview_type;
			typedef aview1d<T>  column_view_type;
			typedef caview1d_ex<T, step_ind> row_cview_type;
			typedef aview1d_ex<T, step_ind>  row_view_type;

			static row_cview_type row_cview(const T* pbase, index_t ldim, index_t d0, index_t d1, index_t irow)
			{
				return caview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, ldim));
			}

			static row_view_type row_view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow)
			{
				return aview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, ldim));
			}

			static column_cview_type column_cview(const T* pbase, index_t ldim, index_t d0, index_t d1, index_t icol)
			{
				return caview1d<T>(pbase + icol * ldim, d0);
			}

			static column_view_type column_view(T* pbase, index_t ldim, index_t d0, index_t d1, index_t icol)
			{
				return aview1d<T>(pbase + icol * ldim, d0);
			}
		};


		template<typename T, typename TOrd, class TRange> struct slice_row_range_helper2d;
		template<typename T, typename TOrd, class TRange> struct slice_col_range_helper2d;

		template<typename T, class TRange>
		struct slice_row_range_helper2d<T, row_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return cview_type(pbase + ldim * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return view_type(pbase + ldim * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}
		};

		template<typename T>
		struct slice_row_range_helper2d<T, row_major_t, whole>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, whole)
			{
				return cview_type(pbase + ldim * irow, d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, whole)
			{
				return view_type(pbase + ldim * irow, d1);
			}
		};

		template<typename T>
		struct slice_row_range_helper2d<T, row_major_t, range>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const range& rgn)
			{
				return cview_type(pbase + ldim * irow + rgn.begin_index(), rgn.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const range& rgn)
			{
				return view_type(pbase + ldim * irow + rgn.begin_index(), rgn.dim());
			}
		};


		template<typename T, class TRange>
		struct slice_col_range_helper2d<T, row_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return cview_type(pbase + ldim * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, ldim, rgn));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return view_type(pbase + ldim * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, ldim, rgn));
			}
		};


		template<typename T, class TRange>
		struct slice_row_range_helper2d<T, column_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return cview_type(pbase + ldim * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, ldim, rgn));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return view_type(pbase + ldim * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, ldim, rgn));
			}
		};

		template<typename T, class TRange>
		struct slice_col_range_helper2d<T, column_major_t, TRange>
		{
			typedef caview1d_ex<T, typename indexer_map<TRange>::type> cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return cview_type(pbase + ldim * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return view_type(pbase + ldim * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}
		};

		template<typename T>
		struct slice_col_range_helper2d<T, column_major_t, whole>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, whole)
			{
				return cview_type(pbase + ldim * icol, d0);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, whole)
			{
				return view_type(pbase + ldim * icol, d0);
			}
		};

		template<typename T>
		struct slice_col_range_helper2d<T, column_major_t, range>
		{
			typedef caview1d<T> cview_type;
			typedef aview1d<T>  view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const range& rgn)
			{
				return cview_type(pbase + ldim * icol + rgn.begin_index(), rgn.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1, index_t icol, const range& rgn)
			{
				return view_type(pbase + ldim * icol + rgn.begin_index(), rgn.dim());
			}
		};
	}


	template<class Derived, typename T, typename TOrd>
	inline typename _detail::slice_helper2d<T, TOrd>::row_cview_type
	row_cview(const IConstBlockAView2D<Derived, T, TOrd>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<T, TOrd> helper;
		return helper::row_cview(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived, typename T, typename TOrd>
	inline typename _detail::slice_helper2d<T, TOrd>::row_view_type
	row_view(IBlockAView2D<Derived, T, TOrd>& a, index_t irow)
	{
		typedef _detail::slice_helper2d<T, TOrd> helper;
		return helper::row_view(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), irow);
	}

	template<class Derived, typename T, typename TOrd>
	inline typename _detail::slice_helper2d<T, TOrd>::column_cview_type
	column_cview(const IConstBlockAView2D<Derived, T, TOrd>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_cview(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), icol);
	}

	template<class Derived, typename T, typename TOrd>
	inline typename _detail::slice_helper2d<T, TOrd>::column_view_type
	column_view(IBlockAView2D<Derived, T, TOrd>& a, index_t icol)
	{
		typedef _detail::slice_helper2d<typename Derived::value_type, typename Derived::layout_order> helper;
		return helper::column_view(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), icol);
	}


	template<class Derived, typename T, typename TOrd, class TRange>
	inline typename _detail::slice_row_range_helper2d<T, TOrd, TRange>::cview_type
	row_cview(const IConstBlockAView2D<Derived, T, TOrd>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<T, TOrd, TRange> helper;
		return helper::cview(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, typename T, typename TOrd, class TRange>
	inline typename _detail::slice_row_range_helper2d<T, TOrd, TRange>::view_type
	row_view(IBlockAView2D<Derived, T, TOrd>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<T, TOrd, TRange> helper;
		return helper::view(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), irow, rgn);
	}

	template<class Derived, typename T, typename TOrd, class TRange>
	inline typename _detail::slice_col_range_helper2d<T, TOrd, TRange>::cview_type
	column_cview(const IConstBlockAView2D<Derived, T, TOrd>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<T, TOrd, TRange> helper;
		return helper::cview(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), icol, rgn);
	}

	template<class Derived, typename T, typename TOrd, class TRange>
	inline typename _detail::slice_col_range_helper2d<T, TOrd, TRange>::view_type
	column_view(IBlockAView2D<Derived, T, TOrd>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<T, TOrd, TRange> helper;
		return helper::view(a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), icol, rgn);
	}


	/******************************************************
	 *
	 *  Sub-views
	 *
	 ******************************************************/

	namespace _detail
	{
		template<typename T, typename TOrd, bool IsCont, class IndexSelector0, class IndexSelector1>
		struct subview_helper2d;
	}

	template<class Derived, typename T, typename TOrd, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::cview_type
	csubview(const IConstBlockAView2D<Derived, T, TOrd>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::cview(
						a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), I, J);
	}

	template<class Derived, typename T, typename TOrd, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::view_type
	subview(IBlockAView2D<Derived, T, TOrd>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::view(
						a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), I, J);
	}


	template<class Derived, typename T, typename TOrd, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::cview_type
	csubview(const IConstContinuousAView2D<Derived, T, TOrd>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::cview(
						a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), I, J);
	}

	template<class Derived, typename T, typename TOrd, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::view_type
	subview(IContinuousAView2D<Derived, T, TOrd>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::view(
						a.pbase(), a.lead_dim(), a.dim0(), a.dim1(), I, J);
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
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class caview2d_ex
	: public IConstRegularAView2D<caview2d_ex<T, TOrd, TIndexer0, TIndexer1>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );

		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	public:
		caview2d_ex(const_pointer pbase, index_type ldim,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(const_cast<pointer>(pbase))
		, m_ldim(ldim)
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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_ldim;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

	protected:
		index_type offset(index_type i, index_type j) const
		{
			return index2d<layout_order>::sub2ind(m_ldim, m_indexer0[i], m_indexer1[j]);
		}

		pointer m_pbase;
		index_t m_ldim;
		index_t m_d0;
		index_t m_d1;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class caview2d_ex



	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	struct aview_traits<aview2d_ex<T, TOrd, TIndexer0, TIndexer1> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
	};

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	class aview2d_ex
	: public caview2d_ex<T, TOrd, TIndexer0, TIndexer1>
	, public IRegularAView2D<aview2d_ex<T, TOrd, TIndexer0, TIndexer1>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer0> );
		BCS_STATIC_ASSERT_V( is_indexer<TIndexer1> );

		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	public:
		aview2d_ex(pointer pbase, index_type ldim,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: caview2d_ex<T, TOrd, TIndexer0, TIndexer1>(pbase, ldim, indexer0, indexer1)
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
			return this->m_d0 * this->m_d1;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return this->m_d0 == 0 || this->m_d1 == 0;
		}

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return this->m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return this->m_d1;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return this->m_d0;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return this->m_d1;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(this->m_d0, this->m_d1);
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return this->m_ldim;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return this->m_pbase[this->offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[this->offset(i, j)];
		}

	}; // end class aview2d_ex


	/******************************************************
	 *
	 *  Block views
	 *
	 ******************************************************/

	template<typename T, typename TOrd>
	struct aview_traits<caview2d_block<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class caview2d_block : public IConstBlockAView2D<caview2d_block<T, TOrd>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		caview2d_block(const_pointer pbase, index_type ldim, index_type m, index_type n)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(ldim), m_d0(m), m_d1(n)
		{
		}

		caview2d_block(const_pointer pbase, index_type ldim, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(ldim), m_d0(shape[0]), m_d1(shape[1])
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
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_d1;
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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_ldim;
		}

	public:
		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[index2d<layout_order>::sub2ind(m_ldim, i, j)];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

	protected:
		pointer m_pbase;
		index_t m_ldim;
		index_t m_d0;
		index_t m_d1;

	}; // end class caview2d_block


	template<typename T, typename TOrd>
	struct aview_traits<aview2d_block<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class aview2d_block
	: public caview2d_block<T, TOrd>
	, public IBlockAView2D<aview2d_block<T, TOrd>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		aview2d_block(pointer pbase, index_type ldim, index_type m, index_type n)
		: caview2d_block<T, TOrd>(pbase, ldim, m, n)
		{
		}

		aview2d_block(pointer pbase, index_type ldim, const shape_type& shape)
		: caview2d_block<T, TOrd>(pbase, ldim, shape)
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
			return this->m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return this->m_d1;
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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return this->m_ldim;
		}

	public:
		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return this->m_pbase;
		}

		BCS_ENSURE_INLINE pointer pbase()
		{
			return this->m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return this->m_pbase[index2d<layout_order>::sub2ind(this->m_ldim, i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[index2d<layout_order>::sub2ind(this->m_ldim, i, j)];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		row_view_type row(index_t i)
		{
			return row_view(*this, i);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		column_view_type column(index_t i)
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
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
			return column_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, false, TRange0, TRange1>::view_type
		V(const TRange0& I, const TRange1& J)
		{
			return subview(*this, I, J);
		}

	}; // end class aview2d_block



	/******************************************************
	 *
	 *  Continuous views
	 *
	 ******************************************************/

	template<typename T, typename TOrd>
	struct aview_traits<caview2d<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class caview2d
	: public IConstContinuousAView2D<caview2d<T, TOrd>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		caview2d(const_pointer pbase, index_type m, index_type n)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(index2d<layout_order>::get_lead_dim(m, n))
		, m_d0(m), m_d1(n)
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(index2d<layout_order>::get_lead_dim(shape[0], shape[1]))
		, m_d0(shape[0]), m_d1(shape[1])
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
			return m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return m_d1;
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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_ldim;
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
			return m_pbase[index2d<layout_order>::sub2ind(m_ldim, i, j)];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		caview1d<T> flatten() const
		{
			return caview1d<T>(pbase(), nelems());
		}

	protected:
		pointer m_pbase;
		index_t m_ldim;
		index_t m_d0;
		index_t m_d1;

	}; // end class caview2d


	template<typename T, typename TOrd>
	struct aview_traits<aview2d<T, TOrd> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)
	};

	template<typename T, typename TOrd>
	class aview2d
	: public caview2d<T, TOrd>
	, public IContinuousAView2D<aview2d<T, TOrd>, T, TOrd>
	{
	public:
		BCS_STATIC_ASSERT_V( is_layout_order<TOrd> );
		BCS_AVIEW_TRAITS_DEFS(2u, T, TOrd)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T, TOrd)

	public:
		aview2d(pointer pbase, index_type m, index_type n)
		: caview2d<T, TOrd>(pbase, m, n)
		{
		}

		aview2d(pointer pbase, const shape_type& shape)
		: caview2d<T, TOrd>(pbase, shape)
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
			return this->m_d0;
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return this->m_d1;
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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return this->m_ldim;
		}

	public:
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return this->m_pbase[index2d<layout_order>::sub2ind(this->m_ldim, i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[index2d<layout_order>::sub2ind(this->m_ldim, i, j)];
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		row_view_type row(index_t i)
		{
			return row_view(*this, i);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		column_view_type column(index_t i)
		{
			return column_view(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, layout_order, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
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
			return column_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, layout_order, TRange>::view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, TOrd, true, TRange0, TRange1>::view_type
		V(const TRange0& I, const TRange1& J)
		{
			return subview(*this, I, J);
		}

		caview1d<T> flatten() const
		{
			return caview1d<T>(pbase(), nelems());
		}

		aview1d<T> flatten()
		{
			return aview1d<T>(pbase(), nelems());
		}

	}; // end class aview2d


	/************************************
	 *
	 * sub-view helper implementation
	 *
	 ************************************/

	namespace _detail
	{
		template<typename T, typename TOrd, bool IsCont, class IndexSelector0, class IndexSelector1>
		struct subview_helper2d
		{
			typedef typename indexer_map<IndexSelector0>::type sindexer0_t;
			typedef typename indexer_map<IndexSelector1>::type sindexer1_t;
			typedef caview2d_ex<T, TOrd, sindexer0_t, sindexer1_t> cview_type;
			typedef aview2d_ex<T, TOrd, sindexer0_t, sindexer1_t> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const IndexSelector0& I, const IndexSelector1& J)
			{
				index_t i0 = indexer_map<IndexSelector0>::get_offset(d0, I);
				index_t j0 = indexer_map<IndexSelector1>::get_offset(d1, J);

				const T *p0 = pbase + index2d<TOrd>::sub2ind(ldim, i0, j0);
				return cview_type(p0, ldim,
						indexer_map<IndexSelector0>::get_indexer(d0, I),
						indexer_map<IndexSelector1>::get_indexer(d1, J));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const IndexSelector0& I, const IndexSelector1& J)
			{
				index_t i0 = indexer_map<IndexSelector0>::get_offset(d0, I);
				index_t j0 = indexer_map<IndexSelector1>::get_offset(d1, J);

				T *p0 = pbase + index2d<TOrd>::sub2ind(ldim, i0, j0);
				return view_type(p0, ldim,
						indexer_map<IndexSelector0>::get_indexer(d0, I),
						indexer_map<IndexSelector1>::get_indexer(d1, J));
			}
		};


		template<typename T, typename TOrd>
		struct subview_helper2d<T, TOrd, false, whole, whole>
		{
			typedef caview2d_block<T, TOrd> cview_type;
			typedef aview2d_block<T, TOrd> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, whole)
			{
				return cview_type(pbase, ldim, d0, d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, whole)
			{
				return view_type(pbase, ldim, d0, d1);
			}
		};

		template<typename T, typename TOrd>
		struct subview_helper2d<T, TOrd, true, whole, whole>
		{
			typedef caview2d<T, TOrd> cview_type;
			typedef aview2d<T, TOrd> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, whole)
			{
				return cview_type(pbase, d0, d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, whole)
			{
				return view_type(pbase, d0, d1);
			}
		};


		template<typename T, typename TOrd, bool IsCont>
		struct subview_helper2d<T, TOrd, IsCont, whole, range>
		{
			typedef caview2d_block<T, TOrd> cview_type;
			typedef aview2d_block<T, TOrd> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return cview_type(pbase + index2d<TOrd>::sub2ind(ldim, 0, J.begin_index()), ldim, d0, J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return view_type(pbase + index2d<TOrd>::sub2ind(ldim, 0, J.begin_index()), ldim, d0, J.dim());
			}
		};


		template<typename T>
		struct subview_helper2d<T, column_major_t, true, whole, range>
		{
			typedef caview2d<T, column_major_t> cview_type;
			typedef aview2d<T, column_major_t> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return cview_type(pbase + d0 * J.begin_index(), d0, J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return view_type(pbase + d0 * J.begin_index(), d0, J.dim());
			}
		};


		template<typename T, typename TOrd, bool IsCont>
		struct subview_helper2d<T, TOrd, IsCont, range, whole>
		{
			typedef caview2d_block<T, TOrd> cview_type;
			typedef aview2d_block<T, TOrd> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return cview_type(pbase + index2d<TOrd>::sub2ind(ldim, I.begin_index(), 0), ldim, I.dim(), d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return view_type(pbase + index2d<TOrd>::sub2ind(ldim, I.begin_index(), 0), ldim, I.dim(), d1);
			}
		};


		template<typename T>
		struct subview_helper2d<T, row_major_t, true, range, whole>
		{
			typedef caview2d<T, row_major_t> cview_type;
			typedef aview2d<T, row_major_t> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return cview_type(pbase + I.begin_index() * d1, I.dim(), d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return view_type(pbase + I.begin_index() * d1, I.dim(), d1);
			}
		};


		template<typename T, typename TOrd, bool IsCont>
		struct subview_helper2d<T, TOrd, IsCont, range, range>
		{
			typedef caview2d_block<T, TOrd> cview_type;
			typedef aview2d_block<T, TOrd> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, const range& J)
			{
				return cview_type(pbase + index2d<TOrd>::sub2ind(ldim, I.begin_index(), J.begin_index()),
						ldim, I.dim(), J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, const range& J)
			{
				return view_type(pbase + index2d<TOrd>::sub2ind(ldim, I.begin_index(), J.begin_index()),
						ldim, I.dim(), J.dim());
			}
		};

	}


	/************************************
	 *
	 *  Convenient functions
	 *
	 ************************************/

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

	template<typename T, class TIndexer0, class TIndexer1>
	inline caview2d_ex<T, row_major_t, TIndexer0, TIndexer1> make_caview2d_ex_rm(
			const T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return caview2d_ex<T, row_major_t, TIndexer0, TIndexer1>(base, index2d<row_major_t>::get_lead_dim(m, n), idx0, idx1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline aview2d_ex<T, row_major_t, TIndexer0, TIndexer1> make_aview2d_ex_rm(
			T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return aview2d_ex<T, row_major_t, TIndexer0, TIndexer1>(base, index2d<row_major_t>::get_lead_dim(m, n), idx0, idx1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline caview2d_ex<T, column_major_t, TIndexer0, TIndexer1> make_caview2d_ex_cm(
			const T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return caview2d_ex<T, column_major_t, TIndexer0, TIndexer1>(base, index2d<row_major_t>::get_lead_dim(m, n), idx0, idx1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline aview2d_ex<T, column_major_t, TIndexer0, TIndexer1> make_aview2d_ex_cm(
			T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return aview2d_ex<T, column_major_t, TIndexer0, TIndexer1>(base, index2d<column_major_t>::get_lead_dim(m, n), idx0, idx1);
	}
}

#endif
