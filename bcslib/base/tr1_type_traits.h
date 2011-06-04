/**
 * @file tr1_utils.h
 *
 * Import of tr1 type traits from TR1 into bcs namespace
 * 
 * @author Dahua Lin
 */

#ifndef BCS_TR1_UTILS_H
#define BCS_TR1_UTILS_H

#include <bcslib/base/config.h>

#if (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_STD_DIR)
#include <type_traits>
#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/type_traits>
#endif

namespace tr1
{
	using BCS_TR1_FROM_NAMESPACE::true_type;
	using BCS_TR1_FROM_NAMESPACE::false_type;

	// primary type categories

	using BCS_TR1_FROM_NAMESPACE::is_void;
	using BCS_TR1_FROM_NAMESPACE::is_integral;
	using BCS_TR1_FROM_NAMESPACE::is_floating_point;
	using BCS_TR1_FROM_NAMESPACE::is_array;
	using BCS_TR1_FROM_NAMESPACE::is_pointer;
	using BCS_TR1_FROM_NAMESPACE::is_reference;
	using BCS_TR1_FROM_NAMESPACE::is_enum;
	using BCS_TR1_FROM_NAMESPACE::is_union;
	using BCS_TR1_FROM_NAMESPACE::is_class;
	using BCS_TR1_FROM_NAMESPACE::is_function;
	using BCS_TR1_FROM_NAMESPACE::is_member_object_pointer;
	using BCS_TR1_FROM_NAMESPACE::is_member_function_pointer;

	// composite type categories

	using BCS_TR1_FROM_NAMESPACE::is_arithmetic;
	using BCS_TR1_FROM_NAMESPACE::is_fundamental;
	using BCS_TR1_FROM_NAMESPACE::is_scalar;
	using BCS_TR1_FROM_NAMESPACE::is_compound;
	using BCS_TR1_FROM_NAMESPACE::is_object;
	using BCS_TR1_FROM_NAMESPACE::is_member_pointer;

	// useful properties

	using BCS_TR1_FROM_NAMESPACE::is_const;
	using BCS_TR1_FROM_NAMESPACE::is_volatile;
	using BCS_TR1_FROM_NAMESPACE::is_pod;
	using BCS_TR1_FROM_NAMESPACE::is_empty;

	using BCS_TR1_FROM_NAMESPACE::is_signed;
	using BCS_TR1_FROM_NAMESPACE::is_unsigned;
	using BCS_TR1_FROM_NAMESPACE::rank;
	using BCS_TR1_FROM_NAMESPACE::extent;

	// type relations

	using BCS_TR1_FROM_NAMESPACE::is_same;
	using BCS_TR1_FROM_NAMESPACE::is_base_of;
	using BCS_TR1_FROM_NAMESPACE::is_convertible;

}

#endif 
