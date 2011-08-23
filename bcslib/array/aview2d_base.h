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

namespace bcs
{

	// extents & index calculation

	struct row_extent
	{
		index_t value;
		explicit row_extent(const index_t& v) : value(v) { }

		BCS_ENSURE_INLINE index_t sub2ind(index_t i, index_t j) const
		{
			return i * value + j;
		}
	};

	struct column_extent
	{
		index_t value;
		explicit column_extent(const index_t& v) : value(v) { }

		BCS_ENSURE_INLINE index_t sub2ind(index_t i, index_t j) const
		{
			return i + value * j;
		}
	};

	template<typename TOrd> struct extent_of;
	template<> struct extent_of<row_major_t> { typedef row_extent type; };
	template<> struct extent_of<column_major_t> { typedef column_extent type; };

	inline BCS_ENSURE_INLINE
	index_t sub2ind(index_t m0, index_t n0, index_t i, index_t j, row_major_t)
	{
		return i * n0 + j;
	}

	inline BCS_ENSURE_INLINE
	index_t sub2ind(index_t m0, index_t n0, index_t i, index_t j, column_major_t)
	{
		return i + m0 * j;
	}

	inline BCS_ENSURE_INLINE
	row_extent get_extent(index_t m0, index_t n0, row_major_t)
	{
		return row_extent(n0);
	}

	inline BCS_ENSURE_INLINE
	column_extent get_extent(index_t m0, index_t n0, column_major_t)
	{
		return column_extent(m0);
	}


	// concept interfaces

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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
		}

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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
		}

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
	class block_caview2d_base
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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		// -- new --

		typename aview_traits<Derived>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

	}; // end class block_caview2d_base


	template<class Derived>
	class block_aview2d_base : public block_caview2d_base<Derived>
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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

		// -- new --

		typename aview_traits<Derived>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::row_view_type
		row(index_t i)
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

		typename aview_traits<Derived>::column_view_type
		column(index_t i)
		{
			return derived().column(i);
		}

	}; // end class block_aview2d_base


	template<class Derived>
	class continuous_caview2d_base
	: public tyselect<aview_traits<Derived>::is_const_view,
	  	  block_caview2d_base<Derived>,
	  	  block_aview2d_base<Derived> >::type
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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
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

		typename aview_traits<Derived>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

		// -- new --

		typename aview_traits<Derived>::flatten_cview_type
		flatten() const
		{
			return derived().flatten();
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

		BCS_ENSURE_INLINE typename extent_of<layout_order>::type base_extent() const
		{
			return derived().base_extent();
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

		typename aview_traits<Derived>::row_cview_type
		row(index_t i) const
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::row_view_type
		row(index_t i)
		{
			return derived().row(i);
		}

		typename aview_traits<Derived>::column_cview_type
		column(index_t i) const
		{
			return derived().column(i);
		}

		typename aview_traits<Derived>::column_view_type
		column(index_t i)
		{
			return derived().column(i);
		}

		// -- new --

		typename aview_traits<Derived>::flatten_cview_type
		flatten() const
		{
			return derived().flatten();
		}

		typename aview_traits<Derived>::flatten_view_type
		flatten()
		{
			return derived().flatten();
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
