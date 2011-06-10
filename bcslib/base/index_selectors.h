/**
 * @file index_selectors.h
 *
 * The classes that implement index selectors for
 * selecting a subset of indices within [0, n-1]
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_INDEX_SELECTORS_H
#define BCSLIB_INDEX_SELECTORS_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/iterator_wrappers.h>

#include <type_traits>
#include <array>
#include <vector>

namespace bcs
{

	/*****
	 *
	 *  The Concept of index selector.
	 *  --------------------------------
	 *
	 *  Let S be an index selector. It should support:
	 *
	 *  - typedefs of the following types:
	 *    - value_type (implicitly convertible to index_t)
	 *    - size_type (size_t)
	 *    - const_iterator: of concept forward_iterator
	 *
	 *  - copy-constructible and/or move-constructible
	 *
	 *  - S.size() -> the number of indices, of type size_type
	 *  - S[i] -> the i-th selected index
	 *  - S.begin() -> the beginning iterator, of type const_iterator
	 *  - S.end() -> the pass-by-end iterator, of type const_iterator
	 *
	 *  We use a trait struct: bcs::index_selector_traits for
	 *  compile-time decision of relevant properties.
	 *
	 *  All these member functions are with const qualification
	 *
	 *  Note that, a series of containers with value_type index_t are in
	 *  this category, such as std::vector<index_t>, std::array<index_t>,
	 *  and std::array1d<index_t>, etc.
	 */


	template<class IndexSelector>
	struct index_selector_traits
	{
		typedef index_t input_type;
	};


	template<typename TInt>
	struct index_selector_traits<std::vector<TInt> >
	{
		BCS_STATIC_ASSERT_V(std::is_integral<TInt>);
		typedef typename std::vector<TInt>::size_type input_type;
	};


	template<typename TInt, size_t N>
	struct index_selector_traits<std::array<TInt, N> >
	{
		BCS_STATIC_ASSERT_V(std::is_integral<TInt>);
		typedef typename std::array<TInt, N>::size_type input_type;
	};



	struct whole { };
	struct rev_whole { };

	namespace _detail
	{
		class _range_iter_impl
		{
		public:
			typedef index_t value_type;
			typedef const index_t& reference;
			typedef const index_t* pointer;

			_range_iter_impl() : m_idx(0) { }

			_range_iter_impl(const index_t& i): m_idx(i) { }

			void move_next() { ++ m_idx; }

			pointer ptr() const { return &m_idx; }

			reference ref() const { return m_idx; }

			bool operator == (const _range_iter_impl& rhs) const
			{
				return m_idx == rhs.m_idx;
			}

		private:
			index_t m_idx;

		}; // end _range_iter_impl


		class _step_range_iter_impl
		{
		public:
			typedef index_t value_type;
			typedef const index_t& reference;
			typedef const index_t* pointer;

			_step_range_iter_impl() : m_idx(0), m_step(1) { }

			_step_range_iter_impl(const index_t& i, const index_t& s): m_idx(i), m_step(s) { }

			void move_next() { m_idx += m_step; }

			pointer ptr() const { return &m_idx; }

			reference ref() const { return m_idx; }

			bool operator == (const _step_range_iter_impl& rhs) const
			{
				return m_idx == rhs.m_idx;
			}

		private:
			index_t m_idx;
			index_t m_step;

		}; // end _step_range_iter_impl


		class _rep_iter_impl
		{
		public:
			typedef index_t value_type;
			typedef const index_t& reference;
			typedef const index_t* pointer;

			_rep_iter_impl() : m_index(0), m_i(0) { }

			_rep_iter_impl(const index_t& idx, const size_t& i): m_index(idx), m_i(i) { }

			void move_next() { ++ m_i; }

			pointer ptr() const { return &m_index; }

			reference ref() const { return m_index; }

			bool operator == (const _rep_iter_impl& rhs) const
			{
				return m_i == rhs.m_i;
			}

		private:
			index_t m_index;
			size_t m_i;

		}; // end _rep_iter_impl
	}


	class range
	{
	public:
		typedef index_t value_type;
		typedef size_t size_type;
		typedef forward_iterator_wrapper<_detail::_range_iter_impl> const_iterator;

	public:
		range()
		: m_begin(0), m_end(0)
		{
		}

		range(index_t b, index_t e)
		: m_begin(b), m_end(e)
		{
		}

		index_t begin_index() const
		{
			return m_begin;
		}

		index_t end_index() const
		{
			return m_end;
		}

		size_t size() const
		{
			return m_begin < m_end ? static_cast<size_t>(m_end - m_begin) : 0;
		}

		index_t operator[] (const index_t& i) const
		{
			return m_begin + i;
		}

		index_t front() const
		{
			return m_begin;
		}

		index_t back() const
		{
			return m_end - 1;
		}

		const_iterator begin() const
		{
			return _detail::_range_iter_impl(begin_index());
		}

		const_iterator end() const
		{
			return _detail::_range_iter_impl(end_index());
		}

		bool operator == (const range& rhs) const
		{
			return m_begin == rhs.m_begin && m_end == rhs.m_end;
		}

		bool operator != (const range& rhs) const
		{
			return !(operator == (rhs));
		}

	private:
		index_t m_begin;
		index_t m_end;

	}; // end class range


	class step_range
	{
	public:
		typedef index_t value_type;
		typedef size_t size_type;
		typedef forward_iterator_wrapper<_detail::_step_range_iter_impl> const_iterator;

	public:
		step_range()
		: m_begin(0), m_n(0), m_step(0)
		{
		}

	private:
		step_range(index_t b, index_t n, index_t step)
		: m_begin(b), m_n(n), m_step(step)
		{
		}

	public:
		static step_range from_begin_end(index_t b, index_t e, index_t step)
		{
			return step_range(b, _calc_n(b, e, step), step);
		}

		static step_range from_begin_dim(index_t b, index_t n, index_t step)
		{
			return step_range(b, n, step);
		}


	public:
		index_t begin_index() const
		{
			return m_begin;
		}

		index_t end_index() const
		{
			return m_begin + m_n * m_step;
		}

		index_t step() const
		{
			return m_step;
		}

		size_t size() const
		{
			return static_cast<size_t>(m_n);
		}

		index_t operator[] (const index_t& i) const
		{
			return m_begin + i * m_step;
		}

		index_t front() const
		{
			return m_begin;
		}

		index_t back() const
		{
			return m_begin + (m_n - 1) * m_step;
		}

		const_iterator begin() const
		{
			return _detail::_step_range_iter_impl(begin_index(), step());
		}

		const_iterator end() const
		{
			return _detail::_step_range_iter_impl(end_index(), step());
		}

		bool operator == (const step_range& rhs) const
		{
			return m_begin == rhs.m_begin && m_n == rhs.m_n && m_step == rhs.m_step;
		}

		bool operator != (const step_range& rhs) const
		{
			return !(operator == (rhs));
		}

	private:
		static index_t _calc_n(index_t b, index_t e, index_t step)
		{
            return step > 0 ?
            		((e > b) ? ((e - b - 1) / step + 1) : 0) :
            		((e < b) ? ((e - b + 1) / step + 1) : 0);
		}

	private:
		index_t m_begin;
		index_t m_n;
		index_t m_step;

	}; // end class step_range


	class rep_selector
	{
	public:
		typedef index_t value_type;
		typedef size_t size_type;
		typedef forward_iterator_wrapper<_detail::_rep_iter_impl> const_iterator;

	public:
		rep_selector()
		: m_index(0), m_n(0)
		{
		}

		rep_selector(index_t index, size_t n)
		: m_index(index), m_n(n)
		{
		}

		index_t index() const
		{
			return m_index;
		}

		size_t size() const
		{
			return static_cast<size_t>(m_n);
		}

		index_t operator[] (const index_t& i) const
		{
			return m_index;
		}

		const_iterator begin() const
		{
			return _detail::_rep_iter_impl(m_index, 0);
		}

		const_iterator end() const
		{
			return _detail::_rep_iter_impl(m_index, m_n);
		}

		bool operator == (const rep_selector& rhs) const
		{
			return m_index == rhs.m_index && m_n == rhs.m_n;
		}

		bool operator != (const rep_selector& rhs) const
		{
			return !(operator == (rhs));
		}

	private:
		index_t m_index;
		size_t m_n;

	}; // end class rep_selector


	// convenient functions for constructing regular selector

	inline range rgn(index_t dim, whole)
	{
		return range(0, dim);
	}

	inline step_range rgn(index_t dim, rev_whole)
	{
		return step_range::from_begin_dim(dim-1, dim, -1);
	}

	inline range rgn(index_t begin_index, index_t end_index)
	{
		return range(begin_index, end_index);
	}

	inline step_range rgn(index_t begin_index, index_t end_index, index_t step)
	{
		return step_range::from_begin_end(begin_index, end_index, step);
	}

	inline range rgn_n(index_t begin_index, index_t n)
	{
		return range(begin_index, begin_index + n);
	}

	inline step_range rgn_n(index_t begin_index, index_t n, index_t step)
	{
		return step_range::from_begin_dim(begin_index, n, step);
	}

	inline rep_selector rep(index_t index, size_t repeat_times)
	{
		return rep_selector(index, repeat_times);
	}


	// make vector of indices from boolean variables or predicate

	template<class IndexSequence, typename ForwardIterator>
	inline void add_found_indices(IndexSequence& seq,
			ForwardIterator first, ForwardIterator last, index_t base_index)
	{
		for (index_t i = base_index; first != last; ++first, ++i)
		{
			if (*first) seq.push_back(i);
		}
	}

	template<class IndexSequence, typename ForwardIterator, typename Predicate>
	inline void add_found_indices(IndexSequence& seq,
			ForwardIterator first, ForwardIterator last, Predicate pred, index_t base_index)
	{
		for (index_t i = base_index; first != last; ++first, ++i)
		{
			if (pred(*first)) seq.push_back(i);
		}
	}

	template<class IndexSequence, typename ForwardIterator, typename ForwardIterator2, typename Predicate>
	inline void add_found_indices(IndexSequence& seq,
			ForwardIterator first, ForwardIterator last, ForwardIterator2 first2, Predicate pred, index_t base_index)
	{
		for (index_t i = base_index; first != last; ++first, ++first2, ++i)
		{
			if (pred(*first, *first2)) seq.push_back(i);
		}
	}


	template<typename ForwardIterator>
	inline std::vector<index_t> find_indices(ForwardIterator first, ForwardIterator last)
	{
		std::vector<index_t> s;
		add_found_indices(s, first, last, 0);
		return s;
	}

	template<typename ForwardIterator, typename Predicate>
	inline std::vector<index_t> find_indices(ForwardIterator first, ForwardIterator last, Predicate pred)
	{
		std::vector<index_t> s;
		add_found_indices(s, first, last, pred, 0);
		return s;
	}

	template<typename ForwardIterator, typename ForwardIterator2, typename Predicate>
	inline std::vector<index_t> find_indices(ForwardIterator first, ForwardIterator last, ForwardIterator2 first2, Predicate pred)
	{
		std::vector<index_t> s;
		add_found_indices(s, first, last, first2, pred, 0);
		return s;
	}


	template<class BoolSequence>
	inline std::vector<index_t> indices(const BoolSequence& B)
	{
		return find_indices(B.begin(), B.end());
	}

	template<class Sequence, typename Predicate>
	inline std::vector<index_t> indices(const Sequence& S, Predicate pred)
	{
		return find_indices(S.begin(), S.end(), pred);
	}


	template<class Sequence, class Sequence2, typename Predicate>
	inline std::vector<index_t> indices(const Sequence& S, const Sequence2& S2, Predicate pred)
	{
		check_arg(S.size() == S2.size(), "The sizes of two sequences are not the same.");
		return find_indices(S.begin(), S.end(), S2.begin(), pred);
	}
}


#endif 
