/**
 * @file aview_base.h
 *
 * Basic definitions for array classes
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_AVIEW_BASE_H
#define BCSLIB_AVIEW_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>

#define BCS_CAVIEW_BASE_DEFS(Derived) \
	static const dim_num_t num_dimensions = aview_traits<Derived>::num_dimensions; \
	typedef typename aview_traits<Derived>::value_type value_type; \
	typedef typename aview_traits<Derived>::layout_order layout_order; \
	typedef typename aview_traits<Derived>::pointer pointer; \
	typedef typename aview_traits<Derived>::const_pointer const_pointer; \
	typedef typename aview_traits<Derived>::reference reference; \
	typedef typename aview_traits<Derived>::const_reference const_reference; \
	typedef typename aview_traits<Derived>::size_type size_type; \
	typedef typename aview_traits<Derived>::index_type index_type; \
	typedef typename aview_traits<Derived>::shape_type shape_type; \
	BCS_ENSURE_INLINE const Derived& derived() const { return *(static_cast<const Derived*>(this)); }

#define BCS_AVIEW_BASE_DEFS(Derived) \
	BCS_CAVIEW_BASE_DEFS(Derived) \
	BCS_ENSURE_INLINE Derived& derived() { return *(static_cast<Derived*>(this)); }

#define BCS_AVIEW_TRAITS_DEFS(nd, T, lorder) \
	static const dim_num_t num_dimensions = nd; \
	typedef array_shape_t<nd> shape_type; \
	typedef T value_type; \
	typedef T* pointer; \
	typedef const T* const_pointer; \
	typedef T& reference; \
	typedef const T& const_reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type; \
	typedef lorder layout_order;

namespace bcs
{
	typedef uint8_t dim_num_t;

	/**********************************
	 *
	 *  array shapes
	 *
	 **********************************/

	template<dim_num_t D>
	struct array_shape_t
	{
		static const dim_num_t num_dimensions = D;

		index_t dim[D];

		dim_num_t ndims() const
		{
			return num_dimensions;
		}

		index_t operator[] (const dim_num_t& i) const
		{
			return dim[i];
		}
	};

	inline bool operator == (const array_shape_t<1>& a, const array_shape_t<1>& b)
	{
		return a[0] == b[0];
	}

	inline bool operator == (const array_shape_t<2>& a, const array_shape_t<2>& b)
	{
		return a[0] == b[0] && a[1] == b[1];
	}

	inline bool operator == (const array_shape_t<3>& a, const array_shape_t<3>& b)
	{
		return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
	}

	template<dim_num_t D>
	inline bool operator != (const array_shape_t<D>& a, const array_shape_t<D>& b)
	{
		return !(a == b);
	}


	inline array_shape_t<1> arr_shape(index_t d0)
	{
		array_shape_t<1> shape;
		shape.dim[0] = d0;
		return shape;
	}

	inline array_shape_t<2> arr_shape(index_t d0, index_t d1)
	{
		array_shape_t<2> shape;
		shape.dim[0] = d0;
		shape.dim[1] = d1;
		return shape;
	}

	inline array_shape_t<3> arr_shape(index_t d0, index_t d1, index_t d2)
	{
		array_shape_t<3> shape;
		shape.dim[0] = d0;
		shape.dim[1] = d1;
		shape.dim[2] = d2;
		return shape;
	}


	/********************************************
	 *
	 *   Array layout order
	 *
	 ********************************************/

	struct layout_1d_t { };
	struct row_major_t { };
	struct column_major_t { };

	template<typename T> struct is_layout_order { static const bool value = false; };

	template<> struct is_layout_order<layout_1d_t> { static const bool value = true; };
	template<> struct is_layout_order<row_major_t> { static const bool value = true; };
	template<> struct is_layout_order<column_major_t> { static const bool value = true; };

	namespace _detail
	{
		template<typename T1, typename T2> struct has_same_layout_order_helper { static const bool value = false; };

		template<> struct has_same_layout_order_helper<layout_1d_t, layout_1d_t> { static const bool value = true; };
		template<> struct has_same_layout_order_helper<row_major_t, row_major_t> { static const bool value = true; };
		template<> struct has_same_layout_order_helper<column_major_t, column_major_t> { static const bool value = true; };
	}

	template<class C1, class C2>
	struct has_same_layout_order
	{
		static const bool value = _detail::has_same_layout_order_helper<
				typename C1::layout_order, typename C2::layout_order>::value;
	};


	/********************************************
	 *
	 *   Array-related concepts
	 *
	 *   Base classes for CRTP
	 *
	 ********************************************/

	template<class Derived> struct aview_traits;

	template<class Derived>
	class caview_base
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

	}; // end class caview_base


	template<class Derived>
	class aview_base : public caview_base<Derived>
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

	}; // end class aview_base


	template<class Derived>
	class dense_caview_base : public aview_traits<Derived>::view_base
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

		BCS_ENSURE_INLINE const_pointer pbase() const
		{
			return derived().pbase();
		}

		BCS_ENSURE_INLINE const_reference operator[](index_type i) const
		{
			return derived().operator[](i);
		}

	}; // end class dense_caview_base


	template<class Derived>
	class dense_aview_base : public dense_caview_base<Derived>
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

	}; // end class dense_aview_base


	/********************************************
	 *
	 *   Generic functions
	 *
	 ********************************************/

	template<class Derived>
	typename Derived::const_pointer begin(const dense_caview_base<Derived>& a)
	{
		return a.pbase();
	}

	template<class Derived>
	typename Derived::pointer begin(dense_aview_base<Derived>& a)
	{
		return a.pbase();
	}

	template<class Derived>
	typename Derived::const_pointer end(const dense_caview_base<Derived>& a)
	{
		return a.pbase() + a.size();
	}

	template<class Derived>
	typename Derived::pointer end(dense_aview_base<Derived>& a)
	{
		return a.pbase() + a.size();
	}


	/********************************************
	 *
	 *   Forward declarations of specific
	 *
	 *   array classes
	 *
	 ********************************************/

	// 1D

	template<class Derived> class caview1d_base;
	template<class Derived> class aview1d_base;
	template<class Derived> class dense_caview1d_base;
	template<class Derived> class dense_aview1d_base;

	template<typename T> class caview1d;
	template<typename T> class aview1d;

	template<typename T, class TIndexer> class caview1d_ex;
	template<typename T, class TIndexer> class aview1d_ex;

	// 2D

	template<class Derived> class caview2d_base;
	template<class Derived> class aview2d_base;
	template<class Derived> class dense_caview2d_base;
	template<class Derived> class dense_aview2d_base;

	template<typename T, typename TOrd> class caview2d;
	template<typename T, typename TOrd> class aview2d;

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1> class caview2d_ex;
	template<typename T, typename TOrd, class TIndexer0, class TIndexer1> class aview2d_ex;

}

#endif
