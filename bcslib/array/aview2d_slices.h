/**
 * @file aview2d_aux.h
 *
 * Auxiliary facilities for 2D Array slicing
 * 
 * @author Dahua Lin
 */

#ifndef AVIEW2D_AUX_H_
#define AVIEW2D_AUX_H_

#include <bcslib/array/aview1d.h>
#include <bcslib/array/aview2d_base.h>

#define BCS_AVIEW2D_SLICE_TYPEDEFS(T) \
	typedef caview1d_ex<T, step_ind> row_cview_type; \
	typedef aview1d_ex<T, step_ind> row_view_type; \
	typedef caview1d<T> column_cview_type; \
	typedef aview1d<T> column_view_type;

namespace bcs
{

	/******************************************************
	 *
	 *  Slice-views
	 *
	 ******************************************************/

	namespace _detail
	{
		template<typename T>
		inline caview1d_ex<T, step_ind> row_cview(const T* pbase, index_t ldim, index_t m, index_t n, index_t irow)
		{
			return caview1d_ex<T, step_ind>(pbase + irow, step_ind(n, ldim));
		}

		template<typename T>
		inline aview1d_ex<T, step_ind> row_view(T *pbase, index_t ldim, index_t m, index_t n, index_t irow)
		{
			return aview1d_ex<T, step_ind>(pbase + irow, step_ind(n, ldim));
		}

		template<typename T>
		inline caview1d<T> column_cview(const T* pbase, index_t ldim, index_t m, index_t n, index_t icol)
		{
			return caview1d<T>(pbase + icol * ldim, m);
		}

		template<typename T>
		inline aview1d<T> column_view(T* pbase, index_t ldim, index_t m, index_t n, index_t icol)
		{
			return aview1d<T>(pbase + icol * ldim, m);
		}


		template<typename T, class TRange> struct slice_row_range_helper2d;
		template<typename T, class TRange> struct slice_col_range_helper2d;

		template<typename T, class TRange>
		struct slice_row_range_helper2d
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
		struct slice_col_range_helper2d
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
		struct slice_col_range_helper2d<T, whole>
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
		struct slice_col_range_helper2d<T, range>
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


	// user functions


	template<class Derived, typename T>
	caview1d_ex<T, step_ind>
	row_cview(const IConstBlockAView2D<Derived, T>& a, index_t irow)
	{
		return _detail::row_cview(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), irow);
	}

	template<class Derived, typename T>
	aview1d_ex<T, step_ind>
	row_view(IBlockAView2D<Derived, T>& a, index_t irow)
	{
		return _detail::row_view(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), irow);
	}

	template<class Derived, typename T>
	caview1d<T>
	column_cview(const IConstBlockAView2D<Derived, T>& a, index_t icol)
	{
		return _detail::column_cview(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), icol);
	}

	template<class Derived, typename T>
	aview1d<T>
	column_view(IBlockAView2D<Derived, T>& a, index_t icol)
	{
		return _detail::column_view(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), icol);
	}


	template<class Derived, typename T, class TRange>
	inline typename _detail::slice_row_range_helper2d<T, TRange>::cview_type
	row_cview(const IConstBlockAView2D<Derived, T>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<T, TRange> helper;
		return helper::cview(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), irow, rgn);
	}

	template<class Derived, typename T, class TRange>
	inline typename _detail::slice_row_range_helper2d<T, TRange>::view_type
	row_view(IBlockAView2D<Derived, T>& a, index_t irow, const TRange& rgn)
	{
		typedef _detail::slice_row_range_helper2d<T, TRange> helper;
		return helper::view(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), irow, rgn);
	}

	template<class Derived, typename T, class TRange>
	inline typename _detail::slice_col_range_helper2d<T, TRange>::cview_type
	column_cview(const IConstBlockAView2D<Derived, T>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<T, TRange> helper;
		return helper::cview(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), icol, rgn);
	}

	template<class Derived, typename T, class TRange>
	inline typename _detail::slice_col_range_helper2d<T, TRange>::view_type
	column_view(IBlockAView2D<Derived, T>& a, index_t icol, const TRange& rgn)
	{
		typedef _detail::slice_col_range_helper2d<T, TRange> helper;
		return helper::view(a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), icol, rgn);
	}

}

#endif 
