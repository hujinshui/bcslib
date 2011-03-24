/**
 * @file vecstat.h
 *
 * Statistics over vectors
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECSTAT_H
#define BCSLIB_VECSTAT_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>

namespace bcslib
{

	// generic stats functions

	template<typename T, typename Accumulator>
	inline void vec_accumulate(Accumulator& accum, size_t n, const T *x)
	{
		for (int i = 0; i < n; ++i)
		{
			accum.take_in(x[i]);
		}
	}

	template<typename T, typename Accumulator>
	inline typename Accumulator::value_type vec_stat(size_t n, const T *x)
	{
		if (n > 0)
		{
			Accumulator accum(x[0]);
			vec_accumulate(accum, n-1, x+1);
			return accum.get_value();
		}
		else
		{
			return Accumulator::empty_value();
		}
	}

	template<typename T, typename Accumulator, typename TInit>
	inline typename Accumulator::value_type vec_stat(size_t n, const T *x, const TInit& x0)
	{
		Accumulator accum(x0);
		vec_accumulate(accum, n, x);
		return accum.get_value();
	}


	// specific stats functions

	template<typename T>
	inline T vec_sum(size_t n, const T *x)
	{
		return vec_stat<T, sum_accumulator<T> >(n, x);
	}

	template<typename T>
	inline T vec_sum(size_t n, const T *x, const T& x0)
	{
		return vec_stat<T, sum_accumulator<T> >(n, x, x0);
	}

	template<typename T>
	inline T vec_prod(size_t n, const T *x)
	{
		return vec_stat<T, prod_accumulator<T> >(n, x);
	}

	template<typename T>
	inline T vec_prod(size_t n, const T *x, const T& x0)
	{
		return vec_stat<T, prod_accumulator<T> >(n, x, x0);
	}

	template<typename T>
	inline T vec_min(size_t n, const T *x)
	{
		return vec_stat<T, min_accumulator<T> >(n, x);
	}

	template<typename T>
	inline T vec_min(size_t n, const T *x, const T& x0)
	{
		return vec_stat<T, min_accumulator<T> >(n, x, x0);
	}

	template<typename T>
	inline T vec_max(size_t n, const T *x)
	{
		return vec_stat<T, max_accumulator<T> >(n, x);
	}

	template<typename T>
	inline T vec_max(size_t n, const T *x, const T& x0)
	{
		return vec_stat<T, max_accumulator<T> >(n, x, x0);
	}

	template<typename T>
	inline std::pair<T, T> vec_minmax(size_t n, const T *x)
	{
		return vec_stat<T, minmax_accumulator<T> >(n, x);
	}

	template<typename T>
	inline std::pair<T, T> vec_minmax(size_t n, const T *x, const T& min0, const T& max0)
	{
		return vec_stat<T, max_accumulator<T> >(n, x, std::make_pair(min0, max0));
	}


	template<typename T>
	inline std::pair<index_t, T> vec_index_min(size_t n, const T *x)
	{
		if (n > 0)
		{
			index_t p = 0;
			T v(x[0]);

			for (size_t i = 1; i < n; ++i)
			{
				if (x[i] < v)
				{
					p = (index_t)i;
					v = x[i];
				}
			}

			return std::make_pair(p, v);
		}
		else
		{
			throw empty_accumulation("Cannot take minimum over an empty collection.");
		}
	}


	template<typename T>
	inline std::pair<index_t, T> vec_index_max(size_t n, const T *x)
	{
		if (n > 0)
		{
			index_t p = 0;
			T v(x[0]);

			for (size_t i = 1; i < n; ++i)
			{
				if (x[i] > v)
				{
					p = (index_t)i;
					v = x[i];
				}
			}

			return std::make_pair(p, v);
		}
		else
		{
			throw empty_accumulation("Cannot take maximum over an empty collection.");
		}
	}


	//


}

#endif 
