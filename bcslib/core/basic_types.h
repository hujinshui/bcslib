/**
 * @file basic_defs.h
 *
 * Library-wide basic definitions
 *
 * @author dhlin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BASIC_TYPES_H
#define BCSLIB_BASIC_TYPES_H

#include <bcslib/config/config.h>

#include <cstddef>

#ifdef BCS_USE_C11_STDLIB
#include <cstdint>
#else
#include <tr1/cstdint>
#endif
#include <utility>

// This is for temporary use,
// and will be replaced with nullptr when it is available in most compilers


#define BCS_NULL NULL

namespace bcs
{
	using BCS_TR1::int8_t;
	using BCS_TR1::int16_t;
	using BCS_TR1::int32_t;
	using BCS_TR1::int64_t;

	using BCS_TR1::uint8_t;
	using BCS_TR1::uint16_t;
	using BCS_TR1::uint32_t;
	using BCS_TR1::uint64_t;

	using std::ptrdiff_t;
	using std::size_t;

	typedef ptrdiff_t index_t;

	using std::pair;
	using std::make_pair;
}


#endif



