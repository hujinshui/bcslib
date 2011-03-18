/**
 * @file basic_funcs.h
 *
 * Definitions of some basic functions
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_BASIC_FUNCS_H
#define BCSLIB_BASIC_FUNCS_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>

#include <utility>

namespace bcs
{

	// min

	template<typename T>
	inline const T& min(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	inline T& min(T& a, T& b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	inline const T& min(const T& a, const T& b, const T& c)
	{
		return min(min(a, b), c);
	}

	template<typename T>
	inline T& min(T& a, T& b, T& c)
	{
		return min(min(a, b), c);
	}

	template<typename T>
	inline const T& min(const T& a, const T& b, const T& c, const T& d)
	{
		return min(min(a, b), min(c, d));
	}

	template<typename T>
	inline T& min(T& a, T& b, T& c, T& d)
	{
		return min(min(a, b), min(c, d));
	}

	// max

	template<typename T>
	inline const T& max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	inline T& max(T& a, T& b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	inline const T& max(const T& a, const T& b, const T& c)
	{
		return max(max(a, b), c);
	}

	template<typename T>
	inline T& max(T& a, T& b, T& c)
	{
		return max(max(a, b), c);
	}

	template<typename T>
	inline const T& max(const T& a, const T& b, const T& c, const T& d)
	{
		return max(max(a, b), max(c, d));
	}

	template<typename T>
	inline T& max(T& a, T& b, T& c, T& d)
	{
		return max(max(a, b), max(c, d));
	}

	// min_max

	template<typename T>
	inline std::pair<const T&, const T&> min_max(const T& a, const T& b)
	{
		typedef std::pair<const T&, const T&> pair_t;
		return a < b ? pair_t(a, b) : pair_t(b, a);
	}

	template<typename T>
	inline std::pair<T&, T&> min_max(T& a, T& b)
	{
		typedef std::pair<T&, T&> pair_t;
		return a < b ? pair_t(a, b) : pair_t(b, a);
	}

	template<typename T>
	inline std::pair<const T&, const T&> min_max(const T& a, const T& b, const T& c)
	{
		typedef std::pair<const T&, const T&> pair_t;
		pair_t ab = min_max(a, b);
		return pair_t(min(ab.first, c), max(ab.second, c));
	}

	template<typename T>
	inline std::pair<T&, T&> min_max(T& a, T& b, T& c)
	{
		typedef std::pair<T&, T&> pair_t;
		pair_t ab = min_max(a, b);
		return pair_t(min(ab.first, c), max(ab.second, c));
	}

	// ssort

	template<typename T>
	inline void ssort(T& a, T& b)
	{
		if (a > b)
		{
			T c(a);
			a = b;
			b = c;
		}
	}

	template<typename T>
	inline void ssort(T& a, T& b, T& c)
	{
		ssort(a, b);
		ssort(b, c);
		ssort(a, b);
	}


	// copy_n

	template<typename InputIter, typename OutputIter>
	void copy_n(InputIter src, size_t n, OutputIter dst)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*dst = *src;
			++ src;
			++ dst;
		}
	}

	template<typename Iter>
	size_t count(Iter first, Iter last)
	{
		size_t c = 0;
		while(first != last)
		{
			++ c;
			++ first;
		}
		return c;
	}


}

#endif
