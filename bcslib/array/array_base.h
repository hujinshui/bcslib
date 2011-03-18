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



