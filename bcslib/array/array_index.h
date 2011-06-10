/**
 * @file array_index.h
 *
 * The indexer classes for array element accessing
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_INDEX_H
#define BCSLIB_ARRAY_INDEX_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/index_selectors.h>

namespace bcs
{
	/**
	 * The concept of indexer
	 * ----------------------
	 *
	 * Let I be an indexer, it should support
	 *
	 * - I.size();	// the number of elements along the dimension
	 * - I[i];		// the mapped index of i
	 *
	 * We note that the class indices (defined in index_selection.h)
	 * is already an indexer class.
	 *
	 */

	class id_ind
	{
	public:
		id_ind(size_t n) : m_n(n)
		{
		}

		size_t size() const
		{
			return m_n;
		}

		index_t operator[] (index_t i) const
		{
			return i;
		}

	private:
		size_t m_n;

	}; // end class id_ind


	class step_ind
	{
	public:
		step_ind(size_t n, index_t s) : m_n(n), m_step(s)
		{
		}

		size_t size() const
		{
			return m_n;
		}

		index_t step() const
		{
			return m_step;
		}

		index_t operator[] (index_t i) const
		{
			return i * m_step;
		}

	private:
		size_t m_n;
		index_t m_step;

	}; // end class step_ind


	class rep_ind
	{
	public:
		rep_ind(size_t n) : m_n(n)
		{
		}

		size_t size() const
		{
			return m_n;
		}

		index_t operator[] (index_t i) const
		{
			return 0;
		}

	private:
		size_t m_n;

	}; // end class rep_ind



	// step injection

	// Note: a fully generic version is given in array1d.h
	template<class TIndexer> struct inject_step;

	template<>
	struct inject_step<id_ind>
	{
		typedef step_ind type;

		static type get(const id_ind& idx0, index_t step)
		{
			return step_ind(idx0.size(), step);
		}
	};


	template<>
	struct inject_step<step_ind>
	{
		typedef step_ind type;

		static type get(const step_ind& idx0, index_t step)
		{
			return step_ind(idx0.size(), idx0.step() * step);
		}
	};


	template<>
	struct inject_step<rep_ind>
	{
		typedef rep_ind type;

		static type get(const rep_ind& idx0, index_t step)
		{
			return idx0;
		}
	};


	/****************************************************************
	 *
	 * sub-indexer by imposing a selector upon an indexer
	 *
	 ****************************************************************/

	// Note: a fully generic version is given in array1d.h
	template<class TIndexer, class TSelector> struct sub_indexer;

	// generic specialization when TSelector is whole or rep_selector

	template<class TIndexer>
	struct sub_indexer<TIndexer, whole>
	{
		typedef TIndexer type;

		static type get(const TIndexer& base_indexer, whole, index_t& offset)
		{
			offset = 0;
			return base_indexer;
		}
	};

	template<class TIndexer>
	struct sub_indexer<TIndexer, rep_selector>
	{
		typedef rep_ind type;
		static type get(const TIndexer& base_indexer, const rep_selector& selector, index_t& offset)
		{
			offset = base_indexer[selector.index()];
			return rep_ind(selector.size());
		}
	};


	// specialized versions

	template<>
	struct sub_indexer<id_ind, range>
	{
		typedef id_ind type;

		static type get(const id_ind& base_indexer, const range& rg, index_t& offset)
		{
			offset = rg.begin_index();
			return id_ind(rg.size());
		}
	};

	template<>
	struct sub_indexer<id_ind, step_range>
	{
		typedef step_ind type;

		static type get(const step_ind& base_indexer, const step_range& rg, index_t& offset)
		{
			offset = rg.begin_index();
			return step_ind(rg.size(), rg.step());
		}
	};

	template<>
	struct sub_indexer<step_ind, range>
	{
		typedef step_ind type;

		static type get(const step_ind& base_indexer, const range& rg, index_t& offset)
		{
			offset = rg.begin_index() * base_indexer.step();
			return step_ind(rg.size(), base_indexer.step());
		}
	};

	template<>
	struct sub_indexer<step_ind, step_range>
	{
		typedef step_ind type;

		static type get(const step_ind& base_indexer, const step_range& rg, index_t& offset)
		{
			offset = rg.begin_index() * base_indexer.step();
			return step_ind(rg.size(), rg.step() * base_indexer.step());
		}
	};

}

#endif 
