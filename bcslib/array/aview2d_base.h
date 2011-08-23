/**
 * @file aview2d_base.h
 *
 * The basic concepts for 2D array views
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_AVIEW2D_BASE_H_
#define BCSLIB_AVIEW2D_BASE_H_

#include <bcslib/array/aview_base.h>
#include <bcslib/array/aindex.h>
#include <bcslib/array/details/aview2d_details.h>

namespace bcs
{
	template<class Derived>
	class caview2d_base
	: public tyselect<aview_traits<Derived>::is_continuous,
	  	  typename tyselect<aview_traits<Derived>::is_const_view,
	  	  	  continuous_caview_base<Derived>,
	  	  	  continuous_aview_base<Derived> >::type,
	  	  typename tyselect<aview_traits<Derived>::is_const_view,
	  	  	  caview_base<Derived>,
	  	  	  aview_base<Derived> >::type>::type
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

		// -- new --

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

	}; // end class caview2d_base


	template<class Derived>
	class aview2d_base : public caview2d_base<Derived>
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

		// -- new --

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

	}; // end class aview2d_base


	template<class Derived>
	class dense_caview2d_base
	: public tyselect<aview_traits<Derived>::is_const_view,
	  	  caview2d_base<Derived>,
	  	  aview2d_base<Derived> >::type
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

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		// -- new --

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

	}; // end class dense_caview2d_base


	template<class Derived>
	class dense_aview2d_base : public dense_caview2d_base<Derived>
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

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		// -- new --

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

	}; // end class dense_aview2d_base


	template<class Derived>
	class continuous_caview2d_base
	: public tyselect<aview_traits<Derived>::is_const_view,
	  	  dense_caview2d_base<Derived>,
	  	  dense_aview2d_base<Derived> >::type
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

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return derived().operator[](i);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		// -- new --

		typename _detail::slice_helper2d<value_type, layout_order>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::row_range_cview_type
		row(index_t i, const IndexSelector& rgn) const
		{
			return derived().row(i, rgn);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::column_range_cview_type
		column(index_t i, const IndexSelector& rgn) const
		{
			return derived().column(i, rgn);
		}

		template<class IndexSelector0, class IndexSelector1>
		caview2d_ex<value_type, layout_order,
			typename indexer_map<IndexSelector0>::type,
			typename indexer_map<IndexSelector1>::type>
		V(const IndexSelector0& I, const IndexSelector1& J) const
		{
			return derived().V(I, J);
		}

	}; // end class continuous_caview2d_base


	template<class Derived>
	class continuous_aview2d_base : public continuous_caview2d_base<Derived>
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

		BCS_ENSURE_INLINE index_type dim0() const
		{
			return derived().dim0();
		}

		BCS_ENSURE_INLINE index_type dim1() const
		{
			return derived().dim1();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

		// -- new --

		typename _detail::slice_helper2d<value_type, layout_order>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::row_view_type
		row(index_t i)
		{
			return derived().row(i);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::row_range_cview_type
		row(index_t i, const IndexSelector& rgn) const
		{
			return derived().row(i, rgn);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::row_range_view_type
		row(index_t i, const IndexSelector& rgn)
		{
			return derived().row(i, rgn);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

		typename _detail::slice_helper2d<value_type, layout_order>::column_view_type
		column(index_t i)
		{
			return derived().column(i);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::column_range_cview_type
		column(index_t i, const IndexSelector& rgn) const
		{
			return derived().column(i, rgn);
		}

		template<class IndexSelector>
		typename _detail::slice_range_helper2d<value_type, layout_order, IndexSelector>::column_range_view_type
		column(index_t i, const IndexSelector& rgn)
		{
			return derived().column(i, rgn);
		}

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

	}; // end class continuous_aview2d_base


	// convenient generic functions

	template<class LDerived, class RDerived>
	inline bool is_same_shape(const caview2d_base<LDerived>& lhs, const caview2d_base<RDerived>& rhs)
	{
		return lhs.dim0() == rhs.dim0() && lhs.dim1() == rhs.dim1();
	}

}

#endif 
