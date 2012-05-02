/**
 * @file vecproc_schemes.h
 *
 * The scheme to process vectors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECPROC_SCHEMES_H_
#define BCSLIB_VECPROC_SCHEMES_H_

#include <bcslib/core/basic_defs.h>

namespace bcs
{

	// per-scalar operations
	struct vecscheme_by_scalars
	{
		const index_t length;
		vecscheme_by_scalars(const index_t len) : length(len) { }
	};

	template<int N> struct vecscheme_by_fixed_num_scalars { };


}

#endif /* VECPROC_SCHEMES_H_ */
