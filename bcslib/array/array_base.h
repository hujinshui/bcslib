/**
 * @file array_base.h
 *
 * Basic definitions for array classes
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY_BASE_H
#define BCSLIB_ARRAY_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/basic_mem.h>

#include <type_traits>
#include <complex>
#include <array>
#include <string>

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
	typedef std::array<index_t, nd> shape_type; \
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

	inline std::array<index_t, 1> arr_shape(index_t n)
	{
		std::array<index_t, 1> shape;
		shape[0] = n;
		return shape;
	}

	inline std::array<index_t, 2> arr_shape(index_t d0, index_t d1)
	{
		std::array<index_t, 2> shape;
		shape[0] = d0;
		shape[1] = d1;
		return shape;
	}

	inline std::array<index_t, 3> arr_shape(index_t d0, index_t d1, index_t d2)
	{
		std::array<index_t, 3> shape;
		shape[0] = d0;
		shape[1] = d1;
		shape[2] = d2;
		return shape;
	}

	inline std::array<index_t, 4> arr_shape(index_t d0, index_t d1, index_t d2, index_t d3)
	{
		std::array<index_t, 4> shape;
		shape[0] = d0;
		shape[1] = d1;
		shape[2] = d2;
		shape[3] = d3;
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

	template<typename T> struct is_layout_order : public std::false_type { };

	template<> struct is_layout_order<layout_1d_t> : public std::true_type { };
	template<> struct is_layout_order<row_major_t> : public std::true_type { };
	template<> struct is_layout_order<column_major_t> : public std::true_type { };


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

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
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

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		// -- new --

		void import_from(const_pointer src)
		{
			derived().import_from(src);
		}

		void fill(const value_type& v)
		{
			derived().fill(v);
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

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
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

		void export_to(pointer dst) const
		{
			derived().export_to(dst);
		}

		void import_from(const_pointer src)
		{
			derived().import_from(src);
		}

		void fill(const value_type& v)
		{
			derived().fill(v);
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
	template<typename T, class Alloc=aligned_allocator<T> > class array1d;

	template<typename T, class TIndexer> class caview1d_ex;
	template<typename T, class TIndexer> class aview1d_ex;

	// 2D

	template<class Derived> class caview2d_base;
	template<class Derived> class aview2d_base;
	template<class Derived> class dense_caview2d_base;
	template<class Derived> class dense_aview2d_base;

	template<typename T, typename TOrd> class caview2d;
	template<typename T, typename TOrd> class aview2d;
	template<typename T, typename TOrd, class Alloc=aligned_allocator<T> > class array2d;

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1> class caview2d_ex;
	template<typename T, typename TOrd, class TIndexer0, class TIndexer1> class aview2d_ex;


	/********************************************
	 *
	 *  meta-programming helpers
	 *
	 ********************************************/

	// is_valid_array_value

	template<typename T>
	struct is_valid_array_value
	{
		static const bool value = std::is_pod<T>::value;
	};

	template<typename T>
	struct is_valid_array_value<std::complex<T> >
	{
		static const bool value = std::is_arithmetic<T>::value;
	};

	template<typename T1, typename T2>
	struct is_valid_array_value<std::pair<T1, T2> >
	{
		static const bool value = is_valid_array_value<T1>::value && is_valid_array_value<T2>::value;
	};

	template<typename T, size_t D>
	struct is_valid_array_value<std::array<T, D> >
	{
		static const bool value = is_valid_array_value<T>::value;
	};
}

#endif
