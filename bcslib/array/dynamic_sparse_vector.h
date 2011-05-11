/*
 * @file dynamic_ordered_sparse_vector.h
 *
 * The class to implement a dynamically sparsed vector
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_DYNAMIC_SPARSE_VECTOR_H
#define BCSLIB_DYNAMIC_SPARSE_VECTOR_H

#include <bcslib/array/array_base.h>
#include <vector>
#include <functional>
#include <algorithm>

namespace bcs
{

	template<typename T, typename TComp=std::greater<T> >
	class dynamic_ordered_spvec
	{
	public:
		typedef T value_type;
		typedef index_t index_type;
		typedef size_t size_type;
		typedef TComp comparer_type;
		typedef indexed_entry<value_type, index_type> pair_type;

		typedef std::vector<index_type, aligned_allocator<value_type> > index_container;
		typedef std::vector<value_type, aligned_allocator<value_type> > value_container;

	public:
		template<typename IndIter, typename ValIter>
		dynamic_ordered_spvec(size_type n, size_type len,
				IndIter inds, ValIter vals, value_type thres=value_type())
		: m_d0((index_type)n), m_na(0), m_thres(thres), m_comp()
		, m_sp_inds(new index_container())
		, m_sp_vals(new value_container())
		{
			_init_structure(len, inds, vals);
		}

		template<typename IndIter, typename ValIter>
		dynamic_ordered_spvec(size_type n, size_type len,
				IndIter inds, ValIter vals, comparer_type comp, value_type thres)
		: m_d0((index_type)n), m_na(0), m_thres(thres), m_comp(comp)
		, m_sp_inds(new index_container())
		, m_sp_vals(new value_container())
		{
			_init_structure(len, inds, vals);
		}

	public:
		index_type dim0() const
		{
			return m_d0;
		}

		size_type nactives() const
		{
			return m_na;
		}

		value_type threshold() const
		{
			return m_thres;
		}

		bool can_be_active_value(const value_type& v) const
		{
			return m_comp(v, m_thres);
		}

		const index_type* active_indices() const
		{
			return m_pinds;
		}

		const value_type* active_values() const
		{
			return m_pvals;
		}

		index_type active_index(size_type i) const
		{
			return m_pinds[i];
		}

		value_type active_value(size_type i) const
		{
			return m_pvals[i];
		}

		pair_type active_pair(size_type i) const
		{
			pair_type pa;
			pa.value = active_value(i);
			pa.index = active_index(i);
			return pa;
		}

		size_type position_of_index(index_type k) const  // return na if not found
		{
			for(size_type i = 0; i < m_na; ++i)
			{
				if (active_index(i) == k) return i;
			}
			return nactives();
		}

		bool has_index(index_type k) const
		{
			return position_of_index(k) < nactives();
		}

	public:
		size_type set_value(const index_type& k, const value_type& v)
		{
			size_type i = position_of_index(k);
			if (i < nactives())
			{
				return update_value_at_position(i, v);
			}
			else
			{
				return add_new_pair(k, v);
			}
		}

		size_type update_value_at_position(size_type i, const value_type& v) // i in [0, na)
		{
			value_type v0 = active_value(i);

			if (m_comp(v, v0))
			{
				return increase_value_at_position(i, v);
			}
			else if (m_comp(v0, v))
			{
				return decrease_value_at_position(i, v);
			}
			else
			{
				m_pvals[i] = v;
				return i;
			}
		}

		size_type add_new_pair(const index_type& k, const value_type& v) // k really did not exist
		{
			if (can_be_active_value(v))
			{
				m_sp_inds->push_back(k);
				m_sp_vals->push_back(v);
				set_base_pointers();

				++ m_na;

				return increase_value_at_position(m_na-1, v);
			}
			else
			{
				return nactives();
			}
		}


		size_type increase_value_at_position(size_type i, const value_type& v);

		size_type decrease_value_at_position(size_type i, const value_type& v);

	private:

		struct pair_comparer
		{
			pair_comparer(comparer_type c) : comp(c)
			{
			}

			bool operator() (const pair_type& pa1, const pair_type& pa2)
			{
				return comp(pa1.value, pa2.value);
			}

			comparer_type comp;
		};


		template<typename IndIter, typename ValIter>
		void _init_structure(size_type len, IndIter inds, ValIter vals);

		void set_base_pointers()
		{
			index_container& cinds = *m_sp_inds;
			value_container& cvals = *m_sp_vals;

			m_pinds = &(cinds[0]);
			m_pvals = &(cvals[0]);
		}

	private:
		index_type m_d0;
		size_type m_na;
		value_type m_thres;
		comparer_type m_comp;

		shared_ptr<index_container> m_sp_inds;
		shared_ptr<value_container> m_sp_vals;
		index_type *m_pinds;
		value_type *m_pvals;

	}; // end class dynamic_ordered_spvec


	// Some implementation of dynamic_ordered_spvec

	template<typename T, typename TComp>
	typename dynamic_ordered_spvec<T, TComp>::size_type
	dynamic_ordered_spvec<T, TComp>::increase_value_at_position(size_type i, const value_type& v)
	{
		index_type k = m_pinds[i];

		while (i > 0 && m_comp(v, m_pvals[i-1]))
		{
			m_pinds[i] = m_pinds[i-1];
			m_pvals[i] = m_pvals[i-1];
			-- i;
		}

		m_pinds[i] = k;
		m_pvals[i] = v;

		return i;
	}

	template<typename T, typename TComp>
	typename dynamic_ordered_spvec<T, TComp>::size_type
	dynamic_ordered_spvec<T, TComp>::decrease_value_at_position(size_type i, const value_type& v)
	{
		size_type max_i = nactives() - 1;
		index_type k = m_pinds[i];

		if (can_be_active_value(v))
		{
			while (i < max_i && m_comp(m_pvals[i+1], v))
			{
				m_pinds[i] = m_pinds[i+1];
				m_pvals[i] = m_pvals[i+1];
				++ i;
			}

			m_pinds[i] = k;
			m_pvals[i] = v;
		}
		else
		{
			while(i < max_i)
			{
				m_pinds[i] = m_pinds[i+1];
				m_pvals[i] = m_pvals[i+1];
				++ i;
			}

			m_sp_inds->pop_back();
			m_sp_vals->pop_back();
			-- m_na;
		}

		return i;
	}

	template<typename T, typename TComp>
	template<typename IndIter, typename ValIter>
	void dynamic_ordered_spvec<T, TComp>::_init_structure(size_type len, IndIter inds, ValIter vals)
	{
		// put valid ones as pairs

		std::vector<pair_type> pairs;
		pairs.reserve(len);

		for (size_type i = 0; i < len; ++i)
		{
			index_type idx = *(inds++);
			value_type val = *(vals++);

			if (can_be_active_value(val))
			{
				pair_type pa;
				pa.index = idx;
				pa.value = val;

				pairs.push_back(pa);
			}
		}

		m_na = (size_type)(pairs.size());

		// sort them

		std::sort(pairs.begin(), pairs.end(), pair_comparer(m_comp));

		// put sorted ones into internal storage

		index_container& cinds = *m_sp_inds;
		value_container& cvals = *m_sp_vals;

		cinds.reserve(m_na);
		cvals.reserve(m_na);

		for (size_type i = 0; i < m_na; ++i)
		{
			const pair_type& pa = pairs[i];
			cinds.push_back(pa.index);
			cvals.push_back(pa.value);
		}

		// set base pointers
		set_base_pointers();
	}


}

#endif
