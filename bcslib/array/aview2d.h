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

#include <bcslib/array/aview2d_slices.h>

namespace bcs
{

	/******************************************************
	 *
	 *  Sub-views
	 *
	 ******************************************************/

	namespace _detail
	{
		template<typename T, bool IsCont, class IndexSelector0, class IndexSelector1>
		struct subview_helper2d;
	}

	template<class Derived, typename T, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, false, TRange0, TRange1>::cview_type
	csubview(const IConstBlockAView2D<Derived, T>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, false, TRange0, TRange1>::cview(
				a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), I, J);
	}

	template<class Derived, typename T, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, false, TRange0, TRange1>::view_type
	subview(IBlockAView2D<Derived, T>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, false, TRange0, TRange1>::view(
				a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), I, J);
	}


	template<class Derived, typename T, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, true, TRange0, TRange1>::cview_type
	csubview(const IConstContinuousAView2D<Derived, T>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, true, TRange0, TRange1>::cview(
				a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), I, J);
	}

	template<class Derived, typename T, class TRange0, class TRange1>
	inline typename _detail::subview_helper2d<T, true, TRange0, TRange1>::view_type
	subview(IContinuousAView2D<Derived, T>& a, const TRange0& I, const TRange1& J)
	{
		return _detail::subview_helper2d<T, true, TRange0, TRange1>::view(
				a.pbase(), a.lead_dim(), a.nrows(), a.ncolumns(), I, J);
	}


	/******************************************************
	 *
	 *  Extended Views
	 *
	 ******************************************************/

	template<typename T, class TIndexer0, class TIndexer1>
	struct aview_traits<caview2d_ex<T, TIndexer0, TIndexer1> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
	};

	template<typename T, class TIndexer0, class TIndexer1>
	class caview2d_ex
	: public IConstRegularAView2D<caview2d_ex<T, TIndexer0, TIndexer1>, T>
	{
	public:
#ifdef BCS_USE_STATIC_ASSERT
		static_assert( is_indexer<TIndexer0>::value, "TIndexer0 must be an indexer type." );
		static_assert( is_indexer<TIndexer1>::value, "TIndexer1 must be an indexer type." );
#endif

		BCS_AVIEW_TRAITS_DEFS(2u, T)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	public:
		caview2d_ex(const_pointer pbase, index_type ldim,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: m_pbase(const_cast<pointer>(pbase))
		, m_ldim(ldim)
		, m_nrows(indexer0.dim())
		, m_ncols(indexer1.dim())
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
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_nrows == 0 || m_ncols == 0;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_ldim;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_nrows, m_ncols);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return m_indexer0[i] + m_ldim * m_indexer1[j];
		}

	protected:
		pointer m_pbase;
		index_t m_ldim;
		index_t m_nrows;
		index_t m_ncols;
		indexer0_type m_indexer0;
		indexer1_type m_indexer1;

	}; // end class caview2d_ex



	template<typename T, class TIndexer0, class TIndexer1>
	struct aview_traits<aview2d_ex<T, TIndexer0, TIndexer1> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
	};

	template<typename T, class TIndexer0, class TIndexer1>
	class aview2d_ex
	: public caview2d_ex<T, TIndexer0, TIndexer1>
	, public IRegularAView2D<aview2d_ex<T, TIndexer0, TIndexer1>, T>
	{
	public:
#ifdef BCS_USE_STATIC_ASSERT
		static_assert( is_indexer<TIndexer0>::value, "TIndexer0 must be an indexer type." );
		static_assert( is_indexer<TIndexer1>::value, "TIndexer1 must be an indexer type." );
#endif

		BCS_AVIEW_TRAITS_DEFS(2u, T)

		typedef TIndexer0 indexer0_type;
		typedef TIndexer1 indexer1_type;

	private:
		typedef caview2d_ex<T, TIndexer0, TIndexer1> cview_type;

	public:
		aview2d_ex(pointer pbase, index_type ldim,
				const indexer0_type& indexer0, const indexer1_type& indexer1)
		: cview_type(pbase, ldim, indexer0, indexer1)
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
			return cview_type::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return cview_type::is_empty();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return cview_type::nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return cview_type::ncolumns();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return cview_type::lead_dim();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return cview_type::shape();
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

	template<typename T>
	struct aview_traits<caview2d_block<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)
	};

	template<typename T>
	class caview2d_block : public IConstBlockAView2D<caview2d_block<T>, T>
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)

	public:
		caview2d_block(const_pointer pbase, index_type ldim, index_type m, index_type n)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(ldim), m_nrows(m), m_ncols(n)
		{
		}

		caview2d_block(const_pointer pbase, index_type ldim, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase)), m_ldim(ldim), m_nrows(shape[0]), m_ncols(shape[1])
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
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_nrows == 0 || m_ncols == 0;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_ldim;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(nrows(), ncolumns());
		}

	public:
		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return m_pbase;
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return m_pbase[offset(i, j)];
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return i + m_ldim * j;
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::view_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, false, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

	protected:
		pointer m_pbase;
		index_type m_ldim;
		index_type m_nrows;
		index_type m_ncols;

	}; // end class caview2d_block


	template<typename T>
	struct aview_traits<aview2d_block<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)
	};

	template<typename T>
	class aview2d_block
	: public caview2d_block<T>
	, public IBlockAView2D<aview2d_block<T>, T>
	{
	public:

		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)

	private:
		typedef caview2d_block<T> cview_type;

	public:
		aview2d_block(pointer pbase, index_type ldim, index_type m, index_type n)
		: cview_type(pbase, ldim, m, n)
		{
		}

		aview2d_block(pointer pbase, index_type ldim, const shape_type& shape)
		: cview_type(pbase, ldim, shape)
		{
		}

	public:
		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return cview_type::size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return cview_type::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return cview_type::is_empty();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return cview_type::nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return cview_type::ncolumns();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return cview_type::shape();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return cview_type::lead_dim();
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
			return this->m_pbase[this->offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[this->offset(i, j)];
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
		typename _detail::slice_row_range_helper2d<value_type, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::view_type
		row(index_t i, const TRange& rgn)
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, false, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, false, TRange0, TRange1>::view_type
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

	template<typename T>
	struct aview_traits<caview2d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)
	};

	template<typename T>
	class caview2d
	: public IConstContinuousAView2D<caview2d<T>, T>
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)

	public:
		caview2d(const_pointer pbase, index_type m, index_type n)
		: m_pbase(const_cast<pointer>(pbase))
		, m_nrows(m), m_ncols(n)
		{
		}

		caview2d(const_pointer pbase, const shape_type& shape)
		: m_pbase(const_cast<pointer>(pbase))
		, m_nrows(shape[0]), m_ncols(shape[1])
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
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_nrows == 0 || m_ncols == 0;
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return arr_shape(m_nrows, m_ncols);
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
			return m_pbase[offset(i, j)];
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return i + j * m_nrows;
		}

	public:
		row_cview_type row(index_t i) const
		{
			return row_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		column_cview_type column(index_t i) const
		{
			return column_cview(*this, i);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::view_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, true, TRange0, TRange1>::cview_type
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
		index_t m_nrows;
		index_t m_ncols;

	}; // end class caview2d


	template<typename T>
	struct aview_traits<aview2d<T> >
	{
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)
	};

	template<typename T>
	class aview2d
	: public caview2d<T>
	, public IContinuousAView2D<aview2d<T>, T>
	{
	public:
		BCS_AVIEW_TRAITS_DEFS(2u, T)
		BCS_AVIEW2D_SLICE_TYPEDEFS(T)

	private:
		typedef caview2d<T> cview_type;

	public:
		aview2d(pointer pbase, index_type m, index_type n)
		: cview_type(pbase, m, n)
		{
		}

		aview2d(pointer pbase, const shape_type& shape)
		: cview_type(pbase, shape)
		{
		}


	public:
		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return num_dimensions;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return cview_type::size();
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return cview_type::nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return cview_type::is_empty();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return cview_type::nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return cview_type::ncolumns();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return cview_type::lead_dim();
		}

		BCS_ENSURE_INLINE shape_type shape() const
		{
			return cview_type::shape();
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
			return this->m_pbase[this->offset(i, j)];
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return this->m_pbase[this->offset(i, j)];
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
		typename _detail::slice_row_range_helper2d<value_type, TRange>::cview_type
		row(index_t i, const TRange& rgn) const
		{
			return row_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_row_range_helper2d<value_type, TRange>::view_type
		row(index_t i, const TRange& rgn)
		{
			return row_view(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::cview_type
		column(index_t i, const TRange& rgn) const
		{
			return column_cview(*this, i, rgn);
		}

		template<class TRange>
		typename _detail::slice_col_range_helper2d<value_type, TRange>::view_type
		column(index_t i, const TRange& rgn)
		{
			return column_view(*this, i, rgn);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, true, TRange0, TRange1>::cview_type
		V(const TRange0& I, const TRange1& J) const
		{
			return csubview(*this, I, J);
		}

		template<class TRange0, class TRange1>
		typename _detail::subview_helper2d<T, true, TRange0, TRange1>::view_type
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
		template<typename T, bool IsCont, class IndexSelector0, class IndexSelector1>
		struct subview_helper2d
		{
			typedef typename indexer_map<IndexSelector0>::type sindexer0_t;
			typedef typename indexer_map<IndexSelector1>::type sindexer1_t;
			typedef caview2d_ex<T, sindexer0_t, sindexer1_t> cview_type;
			typedef aview2d_ex<T, sindexer0_t, sindexer1_t> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const IndexSelector0& I, const IndexSelector1& J)
			{
				index_t i0 = indexer_map<IndexSelector0>::get_offset(d0, I);
				index_t j0 = indexer_map<IndexSelector1>::get_offset(d1, J);

				const T *p0 = pbase + (i0 + ldim * j0);
				return cview_type(p0, ldim,
						indexer_map<IndexSelector0>::get_indexer(d0, I),
						indexer_map<IndexSelector1>::get_indexer(d1, J));
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const IndexSelector0& I, const IndexSelector1& J)
			{
				index_t i0 = indexer_map<IndexSelector0>::get_offset(d0, I);
				index_t j0 = indexer_map<IndexSelector1>::get_offset(d1, J);

				T *p0 = pbase + (i0 + ldim * j0);
				return view_type(p0, ldim,
						indexer_map<IndexSelector0>::get_indexer(d0, I),
						indexer_map<IndexSelector1>::get_indexer(d1, J));
			}
		};


		template<typename T>
		struct subview_helper2d<T, false, whole, whole>
		{
			typedef caview2d_block<T> cview_type;
			typedef aview2d_block<T> view_type;

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

		template<typename T>
		struct subview_helper2d<T, true, whole, whole>
		{
			typedef caview2d<T> cview_type;
			typedef aview2d<T> view_type;

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


		template<typename T, bool IsCont>
		struct subview_helper2d<T, IsCont, whole, range>
		{
			typedef caview2d_block<T> cview_type;
			typedef aview2d_block<T> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return cview_type(pbase + (ldim * J.begin_index()), ldim, d0, J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return view_type(pbase + (ldim * J.begin_index()), ldim, d0, J.dim());
			}
		};


		template<typename T>
		struct subview_helper2d<T, true, whole, range>
		{
			typedef caview2d<T> cview_type;
			typedef aview2d<T> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return cview_type(pbase + ldim * J.begin_index(), d0, J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					whole, const range& J)
			{
				return view_type(pbase + ldim * J.begin_index(), d0, J.dim());
			}
		};


		template<typename T, bool IsCont>
		struct subview_helper2d<T, IsCont, range, whole>
		{
			typedef caview2d_block<T> cview_type;
			typedef aview2d_block<T> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return cview_type(pbase + I.begin_index(), ldim, I.dim(), d1);
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, whole)
			{
				return view_type(pbase + I.begin_index(), ldim, I.dim(), d1);
			}
		};


		template<typename T, bool IsCont>
		struct subview_helper2d<T, IsCont, range, range>
		{
			typedef caview2d_block<T> cview_type;
			typedef aview2d_block<T> view_type;

			static cview_type cview(const T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, const range& J)
			{
				return cview_type(pbase + (I.begin_index() + ldim * J.begin_index()),
						ldim, I.dim(), J.dim());
			}

			static view_type view(T *pbase, index_t ldim, index_t d0, index_t d1,
					const range& I, const range& J)
			{
				return view_type(pbase + (I.begin_index() + ldim * J.begin_index()),
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
	inline caview2d<T> make_caview2d(const T *base, index_t m, index_t n)
	{
		return caview2d<T>(base, m, n);
	}

	template<typename T>
	inline aview2d<T> make_aview2d(T *base, index_t m, index_t n)
	{
		return aview2d<T>(base, m, n);
	}


	template<typename T, class TIndexer0, class TIndexer1>
	inline caview2d_ex<T, TIndexer0, TIndexer1> make_caview2d_ex(
			const T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return caview2d_ex<T, TIndexer0, TIndexer1>(base, m, idx0, idx1);
	}

	template<typename T, class TIndexer0, class TIndexer1>
	inline aview2d_ex<T, TIndexer0, TIndexer1> make_aview2d_ex(
			T *base, index_t m, index_t n, const TIndexer0& idx0, const TIndexer1& idx1)
	{
		return aview2d_ex<T, TIndexer0, TIndexer1>(base, m, idx0, idx1);
	}
}

#endif
