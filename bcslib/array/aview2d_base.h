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
	template<typename T> class caview1d;
	template<typename T> class aview1d;


	// index calculation

	template<typename TOrd> struct index2d;

	template<>
	struct index2d<row_major_t>
	{
		BCS_ENSURE_INLINE static index_t sub2ind(index_t ldim, index_t i, index_t j)
		{
			return i * ldim + j;
		}

		BCS_ENSURE_INLINE static index_t get_lead_dim(index_t m, index_t n)
		{
			return n;
		}
	};


	template<>
	struct index2d<column_major_t>
	{
		BCS_ENSURE_INLINE static index_t sub2ind(index_t ldim, index_t i, index_t j)
		{
			return i + j * ldim;
		}

		BCS_ENSURE_INLINE static index_t get_lead_dim(index_t m, index_t n)
		{
			return m;
		}

	};


	// concept interfaces

	template<class Derived, typename T, typename TOrd>
	class IConstAView2DBase
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

	}; // end class caview2d_base


	template<class Derived, typename T, typename TOrd>
	class IAView2DBase
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
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


	template<class Derived, typename T, typename TOrd>
	class IConstRegularAView2D : public IConstAView2DBase<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

	}; // end class dense_caview2d_base


	template<class Derived, typename T, typename TOrd>
	class IRegularAView2D : public IAView2DBase<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
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


	template<class Derived, typename T, typename TOrd>
	class IConstBlockAView2D : public IConstRegularAView2D<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
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


	template<class Derived, typename T, typename TOrd>
	class IBlockAView2D : public IRegularAView2D<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
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


	template<class Derived, typename T, typename TOrd>
	class IConstContinuousAView2D : public IConstBlockAView2D<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
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

		caview1d<T> flatten() const
		{
			return derived().flatten();
		}

	};


	template<class Derived, typename T, typename TOrd>
	class IContinuousAView2D : public IBlockAView2D<Derived, T, TOrd>
	{
	public:
		BCS_AVIEW_INTERFACE_DEFS(Derived)

		BCS_ENSURE_INLINE dim_num_t ndims() const
		{
			return derived().ndims();
		}

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

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
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

		caview1d<T> flatten() const
		{
			return derived().flatten();
		}

		aview1d<T> flatten()
		{
			return derived().flatten();
		}

	};


	// convenient generic functions

	template<class LDerived, typename LT, class RDerived, typename RT, typename TOrd>
	inline bool is_same_shape(
			const IConstAView2DBase<LDerived, LT, TOrd>& lhs,
			const IConstAView2DBase<RDerived, RT, TOrd>& rhs)
	{
		return lhs.dim0() == rhs.dim0() && lhs.dim1() == rhs.dim1();
	}

}

#endif 
