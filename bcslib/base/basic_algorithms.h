/**
 * @file basic_algorithms.h
 *
 * Provide algorithms in addition to the ones in C++ standard
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_BASIC_ALGORITHMS_H
#define BCSLIB_BASIC_ALGORITHMS_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <type_traits>
#include <algorithm>
#include <iterator>


namespace bcs
{

	// min, max, and minmax

	using std::min;
	using std::max;
	using std::minmax;

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
		auto mxy = minmax(x, y);
		return make_pair(min(mxy.first, z), max(z, mxy.second));
	}

	template<typename T, typename Pred>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, Pred comp)
	{
		auto mxy = minmax(x, y, comp);
		return make_pair(min(mxy.first, z, comp), max(z, mxy.second, comp));
	}

	template<typename T>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, const T& w)
	{
		auto mxy = minmax(x, y);
		auto mzw = minmax(z, w);
		return make_pair(min(mxy.first, mzw.first), max(mzw.second, mxy.second));
	}

	template<typename T, typename Pred>
	inline std::pair<const T&, const T&> minmax(const T& x, const T& y, const T& z, const T& w, Pred comp)
	{
		auto mxy = minmax(x, y, comp);
		auto mzw = minmax(z, w, comp);
		return make_pair(min(mxy.first, mzw.first, comp), max(mzw.second, mxy.second, comp));
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


	// iterator-based algorithms


	template<typename ForwardIterator>
	inline ForwardIterator next(ForwardIterator it)
	{
		return ++it;
	}

	template<typename ForwardIterator>
	inline ForwardIterator next2(ForwardIterator it)
	{
		return ++(++it);
	}

	template<typename BidirectionalIterator>
	inline BidirectionalIterator prev(BidirectionalIterator it)
	{
		return --it;
	}

	template<typename BidirectionalIterator>
	inline BidirectionalIterator prev2(BidirectionalIterator it)
	{
		return --(--it);
	}


	template<typename ForwardIterator>
	typename std::iterator_traits<ForwardIterator>::difference_type
	count_all(ForwardIterator first, ForwardIterator last)
	{
		typename std::iterator_traits<ForwardIterator>::difference_type c(0);
		for (; first != last; ++first) ++c;
		return c;
	}

}


#endif 
