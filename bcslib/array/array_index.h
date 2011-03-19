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




}

#endif 
