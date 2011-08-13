/**
 * @file aview1d_base.h
 *
 * The basic concepts for 1D array views
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_AVIEW1D_BASE_H
#define BCSLIB_AVIEW1D_BASE_H

#include <bcslib/array/aview_base.h>
#include <bcslib/array/aindex.h>

namespace bcs
{
	template<class Derived>
	class caview1d_base : public aview_traits<Derived>::dview_base
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

	}; // end class caview1d_base


	template<class Derived>
	class aview1d_base : public caview1d_base<Derived>
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return derived().operator()(i);
		}

	}; // end class aview1d_base


	template<class Derived>
	class dense_caview1d_base : public aview_traits<Derived>::view_nd_base
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

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return derived().operator[](i);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

		// -- new --

		template<class IndexSelector>
		caview1d_ex<value_type, typename indexer_map<IndexSelector>::type>
		V(const IndexSelector& I) const
		{
			return derived().V(I);
		}

	}; // end class dense_caview1d_base


	template<class Derived>
	class dense_aview1d_base : public dense_caview1d_base<Derived>
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

		BCS_ENSURE_INLINE const_reference operator() (index_type i) const
		{
			return derived().operator()(i);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i)
		{
			return derived().operator()(i);
		}

		// -- new --

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

	}; // end class dense_aview1d_base


}

#endif 
