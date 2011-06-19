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
#include <array>
#include <string>


namespace bcs
{
	typedef uint8_t dim_num_t;

	struct layout_1d_t { };
	struct row_major_t { };
	struct column_major_t { };


	/*********
	 *
	 * The concept of an array class
	 *
	 * Let A be an array class with value_type T and #dims D,
	 *
	 * bcs::is_array_view<A>::value is true
	 * bcs::is_array_view_ndim<A, D>::value is true
	 * the class bcs::array_view_traits<A> should be properly specialized.
	 *
	 * Let a be an instance of A,
	 *
	 * The following operations should be supported:
	 *
	 * get_array_shape(a):  the array shape: shape_type
	 * get_num_elems(a): 	the number of elements -> size_t
	 *
	 * get(a, i), get(a, i, j), ...:  get value of individual elements
	 * set(v, a, i), set(v, a, i, j), ...:  set value of individual elements
	 *
	 * begin(a):  returns the begin iterator of a
	 * end(a):    returns the end iterator of a
	 *
	 * is_dense_view(a):  whether the memory layout is dense
	 * when is_dense_view(a) is true, a should support ptr_base(a);
	 */

	template<class A>
	struct is_array_view
	{
		static const bool value = false;
	};

	template<class A, dim_num_t D>
	struct is_array_view_ndim
	{
		static const bool value = false;
	};

	template<class A>
	struct array_view_traits
	{
		typedef typename A::layout_order layout_order;

		typedef typename A::value_type value_type;
		typedef typename A::size_type size_type;
		typedef typename A::index_type index_type;
		typedef typename A::const_reference const_reference;
		typedef typename A::reference reference;
		typedef typename A::const_pointer const_pointer;
		typedef typename A::pointer pointer;
		typedef typename A::const_iterator const_iterator;
		typedef typename A::iterator iterator;

		typedef typename A::shape_type shape_type;

		static const dim_num_t num_dims = A::num_dims;
	};


	template<class A, dim_num_t K> struct array_dim;


	// make array shape

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

	// Other facilities

	class array_dim_mismatch : public std::invalid_argument
	{
	public:
		array_dim_mismatch()
		: std::invalid_argument("The dimensions of array views are not consistent.")
		{
		}

		array_dim_mismatch(std::string& msg)
		: std::invalid_argument(msg)
		{
		}
	};


	template<typename T>
	struct is_valid_array_element
	{
		static const bool value = std::is_pod<T>::value && std::is_same<T, typename std::remove_const<T>::type>::value;
	};

	template<class Arr1, class Arr2>
	struct is_compatible_aviews
	{
		typedef array_view_traits<Arr1> _trs1;
		typedef array_view_traits<Arr2> _trs2;

		static const bool value = is_array_view<Arr1>::value && is_array_view<Arr2>::value &&
				std::is_same<typename _trs1::value_type, typename _trs2::value_type>::value &&
				array_view_traits<Arr1>::num_dims == array_view_traits<Arr2>::num_dims &&
				std::is_same<typename _trs1::layout_order, typename _trs2::layout_order>::value;
	};

	template<class Arr, class TV>
	struct is_compatible_aview_v
	{
		typedef array_view_traits<Arr> _trs;

		static const bool value = is_array_view<Arr>::value &&
				std::is_convertible<TV, typename _trs::value_type>::value;
	};

	template<typename Arr> struct array_creater; // a helper class for constructing arrays

}


// The macros help defining array classes

#define BCS_ARRAY_CHECK_TYPE(T) \
	static_assert(bcs::is_valid_array_element<T>::value, "T must be a valid element type");

#define BCS_ARRAY_BASIC_TYPEDEFS(nd, T, lorder) \
	static const dim_num_t num_dims = nd; \
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






