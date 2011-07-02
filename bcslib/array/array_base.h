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


namespace bcs
{
	typedef uint8_t dim_num_t;


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

	template<typename TOrd> struct layout_aux2d;

	template<>
	struct layout_aux2d<row_major_t>
	{
		BCS_FORCE_INLINE static index_t offset(index_t d0, index_t d1, index_t i, index_t j)
		{
			return i * d1 + j;
		}

		BCS_FORCE_INLINE static std::array<index_t, 2> ind2sub(index_t d0, index_t d1, index_t idx)
		{
			std::array<index_t, 2> sub;
			sub[0] = idx / d1;
			sub[1] = idx - sub[0] * d1;
			return sub;
		}
	};


	template<>
	struct layout_aux2d<column_major_t>
	{
		BCS_FORCE_INLINE static index_t offset(index_t d0, index_t d1, index_t i, index_t j)
		{
			return i + j * d0;
		}

		BCS_FORCE_INLINE static std::array<index_t, 2> ind2sub(index_t d0, index_t d1, index_t idx)
		{
			std::array<index_t, 2> sub;
			sub[1] = idx / d0;
			sub[0] = idx - sub[1] * d0;
			return sub;
		}
	};



	/*********
	 *
	 *  The concept of an array class
	 *  ------------------------------
	 *
	 *  1. type definitions:
	 *
	 *     - value_type;
	 *     - size_type;
	 *     - index_type;
	 *     - const_reference;
	 *     - reference;
	 *     - const_pointer;
	 *     - pointer;
	 *     - const_iterator;
	 *     - iterator;
	 *     - shape_type;
	 *
	 *  2. static constant numbers:
	 *
	 *     - num_dimensions: dim_num_t
	 *
	 *  3. each instance is constructible with shape parameters,
	 *     copy-constructible and assignable
	 *
	 *  4. non-static member functions:
	 *
	 *     - a(i, j, ...) -> (const) reference: element access
	 *     - a.shape() -> shape_type:  get array shape
	 *     - a.nelems() -> size_type:  number of elements
	 *     - a.pbase() -> (const) pointer:  base address
	 *
	 *  5. is_array_view<Arr>::value is set to true.
	 */

	// forward declaration of all array and array view types, and extended array value types

	// 1D

	template<typename T> class caview1d;
	template<typename T> class aview1d;
	template<typename T, class Alloc=aligned_allocator<T> > class array1d;

	template<typename T, class TIndexer> class caview1d_ex;
	template<typename T, class TIndexer> class aview1d_ex;

	// 2D

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

	// array_type

	template<typename T, dim_num_t D, typename TOrd> struct array_type;

	template<typename T, typename TOrd>
	struct array_type<T, 1, TOrd>
	{
		typedef array1d<T> type;
	};

	template<typename T, typename TOrd>
	struct array_type<T, 2, TOrd>
	{
		typedef array2d<T, TOrd> type;
	};

	// array_cast

	template<class Arr, typename T=typename Arr::value_type, dim_num_t D=Arr::num_dimensions>
	struct array_cast : public array_type<T, D, typename Arr::layout_order> { };

	// is_array_view

	template<class A> struct is_array_view : public std::false_type { };

	template<typename T>
	struct is_array_view<caview1d<T> > : public std::true_type { };

	template<typename T>
	struct is_array_view<aview1d<T> > : public std::true_type { };

	template<typename T, class Alloc>
	struct is_array_view<array1d<T, Alloc> > : public std::true_type { };

	template<typename T, typename TOrd>
	struct is_array_view<caview2d<T, TOrd> > : public std::true_type { };

	template<typename T, typename TOrd>
	struct is_array_view<aview2d<T, TOrd> > : public std::true_type { };

	template<typename T, typename TOrd, class Alloc>
	struct is_array_view<array2d<T, TOrd, Alloc> > : public std::true_type { };

	// is_array_view_ndim

	template<class A, dim_num_t D>
	struct is_array_view_ndim : public std::false_type { };

	template<typename T>
	struct is_array_view_ndim<caview1d<T>, 1> : public std::true_type { };

	template<typename T>
	struct is_array_view_ndim<aview1d<T>, 1> : public std::true_type { };

	template<typename T, class Alloc>
	struct is_array_view_ndim<array1d<T, Alloc>, 1> : public std::true_type { };

	template<typename T, typename TOrd>
	struct is_array_view_ndim<caview2d<T, TOrd>, 2> : public std::true_type { };

	template<typename T, typename TOrd>
	struct is_array_view_ndim<aview2d<T, TOrd>, 2> : public std::true_type { };

	template<typename T, typename TOrd, class Alloc>
	struct is_array_view_ndim<array2d<T, TOrd, Alloc>, 2> : public std::true_type { };


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

	// is_compatible_array_views

	template<class Arr1, class Arr2, class Arr3=nil_type> struct are_compatible_array_views;

	template<class Arr1, class Arr2>
	struct are_compatible_array_views<Arr1, Arr2, nil_type>
	{
		static const bool value =
				is_array_view<Arr1>::value && is_array_view<Arr2>::value &&
				std::is_same<typename Arr1::value_type, typename Arr2::value_type>::value &&
				Arr1::num_dimensions == Arr2::num_dimensions &&
				std::is_same<typename Arr1::layout_order, typename Arr2::layout_order>::value;
	};

	template<class Arr1, class Arr2, class Arr3>
	struct are_compatible_array_views
	{
		static const bool value =
				are_compatible_array_views<Arr1, Arr2>::value &&
				are_compatible_array_views<Arr2, Arr3>::value;
	};

}


// The macros help defining array classes

#define BCS_ARRAY_CHECK_TYPE(T) \
	static_assert(bcs::is_valid_array_value<T>::value, "T must be a valid element type");

#define BCS_ARRAY_BASIC_TYPEDEFS(nd, T, lorder) \
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

#endif






