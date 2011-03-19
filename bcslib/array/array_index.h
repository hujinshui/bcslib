/**
 * @file array_index.h
 *
 * The indexer classes for array element accessing
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_INDEX_H
#define BCSLIB_ARRAY_INDEX_H

#include <bcslib/array/array_base.h>
#include <bcslib/array/index_selection.h>

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



	// sub-indexer by imposing a selector upon an indexer

	template<class TIndexer, class TSelector> struct sub_indexer;


	template<class TIndexer>
	struct sub_indexer<TIndexer, whole>
	{
		typedef id_ind type;
		static type get(const TIndexer& base_indexer, whole, index_t& offset)
		{
			offset = 0;
			return base_indexer;
		}
	};

	template<>
	struct sub_indexer<id_ind, range>
	{
		typedef id_ind type;
		static type get(const id_ind& base_indexer, const range& selector, index_t& offset)
		{
			offset = selector.begin;
			return id_ind(selector.size());
		}
	};

	template<>
	struct sub_indexer<id_ind, step_range>
	{
		typedef step_ind type;
		static type get(const id_ind& base_indexer, const step_range& selector, index_t& offset)
		{
			offset = selector.begin;
			return step_ind(selector.size(), selector.step);
		}
	};

	template<>
	struct sub_indexer<id_ind, open_range>
	{
		typedef id_ind type;
		static type get(const id_ind& base_indexer, const open_range& selector, index_t& offset)
		{
			return sub_indexer<id_ind, range>::get(base_indexer, selector.close(base_indexer.size()), offset);
		}
	};

	template<>
	struct sub_indexer<id_ind, open_step_range>
	{
		typedef step_ind type;
		static type get(const id_ind& base_indexer, const open_step_range& selector, index_t& offset)
		{
			return sub_indexer<id_ind, step_range>::get(base_indexer, selector.close(base_indexer.size()), offset);
		}
	};

	template<>
	struct sub_indexer<id_ind, indices>
	{
		typedef indices type;
		static type get(const id_ind& base_indexer, const indices& selector, index_t& offset)
		{
			offset = 0;
			return selector;
		}
	};


	template<>
	struct sub_indexer<step_ind, range>
	{
		typedef step_ind type;
		static type get(const step_ind& base_indexer, const range& selector, index_t& offset)
		{
			offset = base_indexer.step() * selector.begin;
			return step_ind(selector.size(), base_indexer.step());
		}
	};

	template<>
	struct sub_indexer<step_ind, step_range>
	{
		typedef step_ind type;
		static type get(const step_ind& base_indexer, const step_range& selector, index_t& offset)
		{
			offset = base_indexer.step() * selector.begin;
			return step_ind(selector.size(), base_indexer.step() * selector.step);
		}
	};

	template<>
	struct sub_indexer<step_ind, open_range>
	{
		typedef step_ind type;
		static type get(const step_ind& base_indexer, const open_range& selector, index_t& offset)
		{
			return sub_indexer<step_ind, range>::get(base_indexer, selector.close(base_indexer.size()), offset);
		}
	};

	template<>
	struct sub_indexer<step_ind, open_step_range>
	{
		typedef step_ind type;
		static type get(const step_ind& base_indexer, const open_step_range& selector, index_t& offset)
		{
			return sub_indexer<step_ind, step_range>::get(base_indexer, selector.close(base_indexer.size()), offset);
		}
	};

	template<>
	struct sub_indexer<step_ind, indices>
	{
		typedef indices type;
		static type get(const step_ind& base_indexer, const indices& selector, index_t& offset)
		{
			offset = 0;

			size_t n = selector.size();
			block<index_t>* pb = new block<index_t>(n);
			index_t *dst = pb->pbase();

			index_t s = base_indexer.step();
			for (size_t i = 0; i < n; ++i)
			{
				dst[i] = selector[i] * s;
			}

			return indices(pb, own_t());
		}
	};


	template<>
	struct sub_indexer<indices, whole>
	{
		typedef indices type;
		static type get(const indices& base_indexer, whole, index_t& offset)
		{
			offset = 0;
			return base_indexer;
		}
	};

	template<class TSelector>
	struct sub_indexer<indices, TSelector>
	{
		typedef indices type;
		static type get(const indices& base_indexer, const TSelector& selector, index_t& offset)
		{
			if (!is_open_end_selector(selector))
			{
				index_t n = (index_t)selector.size();
				block<index_t>* pb = new block<index_t>(n);
				index_t *dst = pb->pbase();

				typename TSelector::enumerator etor = selector.get_enumerator();

				for (index_t i = 0; i < n; ++i)
				{
					etor.next();
					dst[i] = base_indexer[etor.get()];
				}

				return indices(pb, own_t());
			}
			else
			{
				return sub_indexer(base_indexer, selector.close(base_indexer.size()), offset);
			}
		}
	};



}

#endif 
