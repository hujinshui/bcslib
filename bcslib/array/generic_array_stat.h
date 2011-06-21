/**
 * @file generic_array_stat.h
 *
 * Generic functions for evaluating array stats
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_GENERIC_ARRAY_STAT_H_
#define BCSLIB_GENERIC_ARRAY_STAT_H_

#include <bcslib/array/array_base.h>
#include <bcslib/array/array_expr_base.h>

#include <bcslib/veccomp/vecstat_functors.h>


namespace bcs
{
	// sum & mean

	// sum

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_uniarr_stat<Arr, vec_sum_ftor> >::type
	sum_arr(const Arr& a)
	{
		return _uniarr_stat<Arr, vec_sum_ftor>::default_evaluate(a);
	}


}


#endif 
