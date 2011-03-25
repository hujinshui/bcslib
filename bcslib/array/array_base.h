/**
 * @file array_base.h
 *
 * Basic definitions for array classes
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_BASE_H
#define BCSLIB_ARRAY_BASE_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>
#include <bcslib/base/basic_mem.h>

#include <memory>

namespace bcs
{

	typedef ptrdiff_t index_t;

	typedef uint8_t dim_num_t;

	struct row_major_t { };
	struct column_major_t { };

	struct index_pair
	{
		index_t i0;
		index_t i1;

		index_pair() : i0(0), i1(0) { }

		index_pair(index_t i0_, index_t i1_) : i0(i0_), i1(i1_) { }

		bool operator == (const index_pair& rhs) const
		{
			return i0 == rhs.i0 && i1 == rhs.i1;
		}

		bool operator != (const index_pair& rhs) const
		{
			return !(operator == (rhs));
		}
	};


	struct index_triple
	{
		index_t i0;
		index_t i1;
		index_t i2;

		index_triple() : i0(0), i1(0), i2(0) { }

		index_triple(index_t i0_, index_t i1_, index_t i2_) : i0(i0_), i1(i1_), i2(i2_) { }

		bool operator == (const index_triple& rhs) const
		{
			return i0 == rhs.i0 && i1 == rhs.i1 && i2 == rhs.i2;
		}

		bool operator != (const index_triple& rhs) const
		{
			return !(operator == (rhs));
		}
	};



	// make array shape

	inline tr1::array<size_t, 1> arr_shape(size_t n)
	{
		tr1::array<size_t, 1> shape;
		shape[0] = n;
		return shape;
	}

	inline tr1::array<size_t, 2> arr_shape(size_t d0, size_t d1)
	{
		tr1::array<size_t, 2> shape;
		shape[0] = d0;
		shape[1] = d1;
		return shape;
	}

	inline tr1::array<size_t, 3> arr_shape(size_t d0, size_t d1, size_t d2)
	{
		tr1::array<size_t, 3> shape;
		shape[0] = d0;
		shape[1] = d1;
		shape[2] = d2;
		return shape;
	}

	inline tr1::array<size_t, 4> arr_shape(size_t d0, size_t d1, size_t d2, size_t d3)
	{
		tr1::array<size_t, 4> shape;
		shape[0] = d0;
		shape[1] = d1;
		shape[2] = d2;
		shape[3] = d3;
		return shape;
	}



	// Exception classes

	class array_exception : public base_exception
	{
	public:
		array_exception(const char *msg) : base_exception(msg) { }
	};

	class array_size_mismatch : public array_exception
	{
	public:
		array_size_mismatch() : array_exception("The sizes of array views are not consistent.") { }
		array_size_mismatch(const char *msg) : array_exception(msg) { }
	};

}

#endif 

// The macros help defining array classes

#define BCS_ARRAY_BASIC_TYPEDEFS(nd, T) \
	static const dim_num_t num_dims = nd; \
	typedef tr1::array<size_t, nd> shape_type; \
	typedef T value_type; \
	typedef T* pointer; \
	typedef const T* const_pointer; \
	typedef T& reference; \
	typedef const T& const_reference; \
	typedef size_t size_type; \
	typedef index_t index_type; \
	typedef index_t difference_type;



