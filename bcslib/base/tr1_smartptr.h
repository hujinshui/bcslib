/**
 * @file tr1_smartptr.h
 *
 * Import of TR1 smart pointers into bcs namespace
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TR1_SMARTPTR_H
#define BCSLIB_TR1_SMARTPTR_H

#include <bcslib/base/config.h>

#if (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_STD_DIR)
#include <memory>
#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/memory>
#endif

namespace tr1
{
	using BCS_TR1_FROM_NAMESPACE::swap;
	using BCS_TR1_FROM_NAMESPACE::weak_ptr;
	using BCS_TR1_FROM_NAMESPACE::shared_ptr;

}

#endif 
