/**
 * @file type_traits.h
 *
 * Import of a subset of TR1 (C++0x) stuff
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_TR1_IMPORTS_H_
#define BCSLIB_TR1_IMPORTS_H_

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>

#ifdef USE_C1X_STDLIB
#include <type_traits>
#include <tuple>
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#define BCS_TR1 std
#else
#include <tr1/type_traits>
#include <tr1/tuple>
#include <tr1/array>
#include <tr1/memory>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#define BCS_TR1 std::tr1
#endif

namespace bcs
{
	// type traits

	using BCS_TR1::is_void;
	using BCS_TR1::is_integral;
	using BCS_TR1::is_floating_point;
	using BCS_TR1::is_array;
	using BCS_TR1::is_pointer;
	using BCS_TR1::is_reference;
	using BCS_TR1::is_member_function_pointer;
	using BCS_TR1::is_member_object_pointer;
	using BCS_TR1::is_enum;
	using BCS_TR1::is_union;
	using BCS_TR1::is_class;
	using BCS_TR1::is_function;

	using BCS_TR1::is_arithmetic;
	using BCS_TR1::is_fundamental;
	using BCS_TR1::is_object;
	using BCS_TR1::is_scalar;
	using BCS_TR1::is_compound;
	using BCS_TR1::is_member_pointer;

	using BCS_TR1::is_const;
	using BCS_TR1::is_volatile;
	using BCS_TR1::is_pod;
	using BCS_TR1::is_empty;
	using BCS_TR1::is_polymorphic;
	using BCS_TR1::is_abstract;

	using BCS_TR1::has_trivial_constructor;
	using BCS_TR1::has_trivial_copy;
	using BCS_TR1::has_trivial_assign;
	using BCS_TR1::has_trivial_destructor;

	using BCS_TR1::is_same;
	using BCS_TR1::is_base_of;

	using BCS_TR1::remove_const;
	using BCS_TR1::remove_volatile;
	using BCS_TR1::remove_cv;
	using BCS_TR1::remove_pointer;
	using BCS_TR1::remove_reference;

	using BCS_TR1::add_const;
	using BCS_TR1::add_volatile;
	using BCS_TR1::add_cv;
	using BCS_TR1::add_pointer;
	using BCS_TR1::add_reference;

	using BCS_TR1::alignment_of;

	// tuple & array

	using BCS_TR1::tuple;
	using BCS_TR1::array;

	// shared_ptr

	using BCS_TR1::shared_ptr;

	// hash container

	using BCS_TR1::unordered_map;
	using BCS_TR1::unordered_multimap;
	using BCS_TR1::unordered_set;
	using BCS_TR1::unordered_multiset;

	using BCS_TR1::hash;

}


#endif

