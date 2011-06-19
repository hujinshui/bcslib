/**
 * @file basic_algorithms.h
 *
 * Provide algorithms in addition to the ones in C++ standard
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif


#ifndef BCSLIB_BASIC_ALGORITHMS_H
#define BCSLIB_BASIC_ALGORITHMS_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace bcs
{

	// min, max, and minmax

	using std::min;
	using std::max;

#if (BCSLIB_COMPILER == BCSLIB_MSVC)
	// MSVC uses a non-standard-conformant way:
	// it returns pair<T, T> instead of pair<const T&, const T&>
	// below is a workaround

	// --- MSVC-WORKAROUND (BEGIN) ---
	
	template<typename T>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y)
	{
		return y < x ? std::pair<const T&, const T&>(y, x) : std::pair<const T&, const T&>(x, y);
	}

	template<typename T, typename Pred>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, Pred comp)
	{
		return comp(y, x) ? std::pair<const T&, const T&>(y, x) : std::pair<const T&, const T&>(x, y);
	}
	
	// --- MSVC-WORKAROUND (END) ---

#else
	using std::minmax;
#endif

	template<typename T>
	inline const T& min(const T& x, const T& y, const T& z)
	{
		return min(min(x, y), z);
	}

	template<typename T, typename Pred>
	inline const T& min(const T& x, const T& y, const T& z, Pred comp)
	{
		return min(min(x, y, comp), z, comp);
	}

	template<typename T>
	inline const T& min(const T& x, const T& y, const T& z, const T& w)
	{
		return min(min(x, y), min(z, w));
	}

	template<typename T, typename Pred>
	inline const T& min(const T& x, const T& y, const T& z, const T& w, Pred comp)
	{
		return min(min(x, y, comp), min(z, w, comp), comp);
	}

	template<typename T>
	inline const T& max(const T& x, const T& y, const T& z)
	{
		return max(max(x, y), z);
	}

	template<typename T, typename Pred>
	inline const T& max(const T& x, const T& y, const T& z, Pred comp)
	{
		return max(max(x, y, comp), z, comp);
	}

	template<typename T>
	inline const T& max(const T& x, const T& y, const T& z, const T& w)
	{
		return max(max(x, y), max(z, w));
	}

	template<typename T, typename Pred>
	const T& max(const T& x, const T& y, const T& z, const T& w, Pred comp)
	{
		return max(max(x, y, comp), max(z, w, comp), comp);
	}

	template<typename T>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z)
	{
		auto mxy = bcs::minmax(x, y);
		return std::pair<const T&, const T&>(min(mxy.first, z), max(z, mxy.second));
	}

	template<typename T, typename Pred>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, Pred comp)
	{
		auto mxy = bcs::minmax(x, y, comp);
		return std::pair<const T&, const T&>(min(mxy.first, z, comp), max(z, mxy.second, comp));
	}

	template<typename T>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, const T& w)
	{
		auto mxy = bcs::minmax(x, y);
		auto mzw = bcs::minmax(z, w);
		return std::pair<const T&, const T&>(min(mxy.first, mzw.first), max(mzw.second, mxy.second));
	}

	template<typename T, typename Pred>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, const T& w, Pred comp)
	{
		auto mxy = bcs::minmax(x, y, comp);
		auto mzw = bcs::minmax(z, w, comp);
		return std::pair<const T&, const T&>(min(mxy.first, mzw.first, comp), max(mzw.second, mxy.second, comp));
	}


	// zip and extract

	template<typename InputIterator1, typename InputIterator2, typename OutputIterator>
	void zip_copy(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator dst)
	{
		for(; first1 != last1; ++first1, ++first2, ++dst)
		{
			*dst = make_pair(*first1, *first2);
		}
	}

	template<typename InputIterator, typename OutputIterator1, typename OutputIterator2>
	void dispatch_copy(InputIterator first, InputIterator last, OutputIterator1 dst1, OutputIterator2 dst2)
	{
		for(; first != last; ++first, ++dst1, ++dst2)
		{
			*dst1 = std::get<0>(*first);
			*dst2 = std::get<1>(*first);
		}
	}

	template<typename InputIterator, typename OutputIterator, size_t I>
	void extract_copy(InputIterator first, InputIterator last, size_constant<I>, OutputIterator dst)
	{
		for(; first != last; ++first, ++dst)
		{
			*dst = std::get<I>(*first);
		}
	}

	// sort

	template<typename T, typename Comp>
	inline void simple_sort(T& x, T& y, Comp comp)
	{
		using std::swap;
		if (comp(y, x)) swap(x, y);
	}

	template<typename T>
	inline void simple_sort(T& x, T& y)
	{
		simple_sort(x, y, std::less<T>());
	}

	template<typename T, typename Comp>
	inline void simple_sort(T& x, T& y, T& z, Comp comp)
	{
		using std::swap;

		if (comp(y, x)) swap(x, y);
		if (comp(z, y))
		{
			swap(y, z);
			if (comp(y, x)) swap(x, y);
		}
	}

	template<typename T>
	inline void simple_sort(T& x, T& y, T& z)
	{
		simple_sort(x, y, z, std::less<T>());
	}

	template<typename T, typename Comp>
	inline void simple_sort(T& x, T& y, T& z, T& w, Comp comp)
	{
		using std::swap;
		simple_sort(x, y, z, comp);

		if (comp(w, z))
		{
			swap(z, w);

			if (comp(z, y))
			{
				swap(y, z);
				if (comp(y, x))
				{
					swap(x, y);
				}
			}
		}
	}


	template<typename T>
	inline void simple_sort(T& x, T& y, T& z, T& w)
	{
		simple_sort(x, y, z, w, std::less<T>());
	}



	template<class Tuple, size_t I, typename Comp=std::less<typename std::tuple_element<I, Tuple>::type> >
	struct tuple_cc
	{
		Comp component_comparer;

		tuple_cc() : component_comparer() { }
		tuple_cc(Comp comp) : component_comparer(comp) { }

		bool operator()(const Tuple& lhs, const Tuple& rhs) const
		{
			return component_comparer(std::get<I>(lhs), std::get<I>(rhs));
		}
	};


	template<typename RandomAccessIterator, size_t I, typename Comp>  // Tuple type can be pair, tuple, and array
	inline void sort_tuples_by_component(
			RandomAccessIterator first,
			RandomAccessIterator last,
			size_constant<I>, Comp comp)
	{
		typedef typename std::iterator_traits<RandomAccessIterator>::value_type tuple_type;
		std::sort(first, last, tuple_cc<tuple_type, I, Comp>(comp));
	}

	template<typename RandomAccessIterator, size_t I>
	inline void sort_tuples_by_component(
			RandomAccessIterator first,
			RandomAccessIterator last,
			size_constant<I>)
	{
		typedef typename std::iterator_traits<RandomAccessIterator>::value_type tuple_type;
		typedef typename std::tuple_element<I, tuple_type>::type element_type;

		sort_tuples_by_component(first, last, size_constant<I>(), std::less<element_type>());
	}


	// accumulation

	template<class Cumulator, typename InputIterator, typename Comb>
	void cumulate(Cumulator& cumulator, InputIterator first, InputIterator last, Comb comb)
	{
		for (; first != last; ++first)
		{
			comb(cumulator, *first);
		}
	}

	template<class Cumulator, typename InputIterator, typename Comb>
	void cumulate_n(Cumulator& cumulator, InputIterator first, size_t n, Comb comb)
	{
		for(size_t i = 0; i < n; ++i, ++first)
		{
			comb(cumulator, *first);
		}
	}

}


#endif 
