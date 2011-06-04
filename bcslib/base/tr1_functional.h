/**
 * @file tr1_functional.h
 *
 * Import of TR1 functional facilities into bcs namespace
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TR1_FUNCTIONAL_H
#define BCSLIB_TR1_FUNCTIONAL_H

#include <bcslib/base/config.h>

#if (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_STD_DIR)
#include <functional>
#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/functional>
#endif

namespace tr1
{
	using BCS_TR1_FROM_NAMESPACE::reference_wrapper;
	using BCS_TR1_FROM_NAMESPACE::ref;
	using BCS_TR1_FROM_NAMESPACE::cref;

	using BCS_TR1_FROM_NAMESPACE::function;
	using BCS_TR1_FROM_NAMESPACE::result_of;
	using BCS_TR1_FROM_NAMESPACE::bind;
	using BCS_TR1_FROM_NAMESPACE::is_bind_expression;
	using BCS_TR1_FROM_NAMESPACE::is_placeholder;
}


#endif 
