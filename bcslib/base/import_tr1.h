/*
 * @file import_tr1.h
 *
 * Import some names from TR1 Libraries
 *
 * @author Dahua Lin
 *
 */

#ifndef BCSLIB_IMPORT_TR1
#define BCSLIB_IMPORT_TR1

#include <bcslib/base/config.h>

#if (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_STD_DIR)
#include <array>
#include <tuple>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>

#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/array>
#include <tr1/tuple>
#include <tr1/memory>
#include <tr1/type_traits>
#include <tr1/unordered_set>
#include <tr1/unordered_map>

#endif


namespace bcs
{
	using BCS_TR1_FROM_NAMESPACE::array;
	using BCS_TR1_FROM_NAMESPACE::tuple;
	using BCS_TR1_FROM_NAMESPACE::shared_ptr;
	using BCS_TR1_FROM_NAMESPACE::is_pod;

	using BCS_TR1_FROM_NAMESPACE::unordered_set;
	using BCS_TR1_FROM_NAMESPACE::unordered_map;

}

#endif /* IMPORT_TR1_H_ */
