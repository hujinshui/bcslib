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

#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/array>
#include <tr1/tuple>
#include <tr1/memory>
#include <tr1/type_traits>


#endif


namespace bcs
{
	using BCS_TR1_FROM_NAMESPACE::array;
	using BCS_TR1_FROM_NAMESPACE::tuple;
	using BCS_TR1_FROM_NAMESPACE::shared_ptr;
	using BCS_TR1_FROM_NAMESPACE::is_pod;

}

#endif /* IMPORT_TR1_H_ */
