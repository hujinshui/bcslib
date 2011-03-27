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

namespace bcs
{

	// specific stats functions

	template<typename T>
	inline T vec_sum(size_t n, const T *x)
	{
		return accumulate_n<sum_accumulator<T>, const T*>(n, x);
	}

	template<typename T>
	inline T vec_sum(size_t n, const T *x, const T& x0)
	{
		return accumulate_n<sum_accumulator<T>, const T*, T>(n, x, x0);
	}

	template<typename T>
	inline T vec_prod(size_t n, const T *x)
	{
		return accumulate_n<prod_accumulator<T>, const T*>(n, x);
	}

	template<typename T>
	inline T vec_prod(size_t n, const T *x, const T& x0)
	{
		return accumulate_n<prod_accumulator<T>, const T*, T>(n, x, x0);
	}

	template<typename T>
	inline T vec_min(size_t n, const T *x)
	{
		return accumulate_n<min_accumulator<T>, const T*>(n, x);
	}

	template<typename T>
	inline T vec_min(size_t n, const T *x, const T& x0)
	{
		return accumulate_n<min_accumulator<T>, const T*, T>(n, x, x0);
	}

	template<typename T>
	inline T vec_max(size_t n, const T *x)
	{
		return accumulate_n<max_accumulator<T>, const T*>(n, x);
	}

	template<typename T>
	inline T vec_max(size_t n, const T *x, const T& x0)
	{
		return accumulate_n<max_accumulator<T>, const T*, T>(n, x, x0);
	}

	template<typename T>
	inline std::pair<T, T> vec_minmax(size_t n, const T *x)
	{
		return accumulate_n<minmax_accumulator<T>, const T*>(n, x);
	}

	template<typename T>
	inline std::pair<T, T> vec_minmax(size_t n, const T *x, const T& min0, const T& max0)
	{
		return accumulate_n<minmax_accumulator<T>, const T*, std::pair<T, T> >(
				n, x, std::make_pair(min0, max0));
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


}

#endif 
