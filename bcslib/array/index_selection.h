/**
 * @file index_selection.h
 *
 * The classes that implement index selectors for
 * selecting a subset of indices within [0, n-1]
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_INDEX_SELECTION_H
#define BCSLIB_INDEX_SELECTION_H

#include <bcslib/array/array_base.h>
#include <vector>

namespace bcs
{
	/**
	 * There are two types of index selectors
	 *
	 * 1. close-end selector, whose end is determined, such as range, step_range, etc.
	 *
	 * 		The Concept of close-end selectors.
	 * 		-----------------------------------
	 *
	 * 		Let s be a close-end selector. It should support:
	 *
	 *		1. s.size();		        // return the number of indices
	 *		2. s.get_enumerator();		// return an enumerator of the indices
	 *		3. s.is_contained_in(n);    // test whether all selected indices are in [0, n-1]
	 *		4. s.is_empty();			// return whether the selected set is empty
	 *
	 * 2. open-end selector, whose end is determined only after the dimension n is
	 *    provided, such as whole, open_range, and open_step_range, etc.
	 *
	 * 		The Concept of open-end selectors.
	 * 		---------------------------------
	 *
	 * 		Let s be an open-end selector. It should support:
	 *
	 * 		1. s.close(n);		// return a close-end selector given dimension n
	 *
	 * In both cases, the selector classes should define a static boolean constant
	 * is_open_end within the class, which equals false for close-end selectors, and
	 * equals true for open-end selectors.
	 *
	 * Given a selector s, we have function is_open_end_selector(s) that
	 * returns this bool value.
	 */


	template<typename TSelector>
	inline bool is_open_end_selector(const TSelector& s)
	{
		return TSelector::is_open_end;
	}



	struct range
	{
		static const bool is_open_end = false;

		index_t begin;
		index_t end;

		range(index_t b, index_t e) : begin(b), end(e)
		{
		}

		size_t size() const
		{
			return end > begin ? (size_t)(end - begin) : 0;
		}

		bool is_empty() const
		{
			return end <= begin;
		}

		bool is_contained_in(size_t n) const
		{
			return begin >= 0 && end <= (index_t)n;
		}

		bool operator == (const range& rhs) const
		{
			return begin == rhs.begin && end == rhs.end;
		}

		bool operator != (const range& rhs) const
		{
			return !(operator == (rhs));
		}

		class enumerator
		{
		public:
			typedef index_t value_type;

			enumerator(index_t b, index_t e) : m_e(e), m_i(b-1) { }

			bool next()
			{
				++ m_i;
				return m_i < m_e;
			}

			value_type get() const
			{
				return m_i;
			}

		private:
			index_t m_e;
			index_t m_i;
		};

		enumerator get_enumerator() const
		{
			return enumerator(begin, end);
		}

	}; // end struct range



	struct step_range
	{
		static const bool is_open_end = false;

		index_t begin;
		index_t end;
		index_t step;

		step_range(index_t b, index_t e, index_t s) : begin(b), end(e), step(s)
		{
		}

		size_t size() const
		{
			return step > 0 ?
					((end > begin) ? ((end - begin - 1) / step + 1) : 0) :
					((end < begin) ? ((end - begin + 1) / step + 1) : 0);
		}

		bool is_empty() const
		{
			return step > 0 ? end <= begin : end >= begin;
		}

		bool is_contained_in(size_t n) const
		{
			return step > 0 ? (begin >= 0 && end <= (index_t)n) : (begin < (index_t)n && end >= 0);
		}

		bool operator == (const step_range& rhs) const
		{
			return begin == rhs.begin && end == rhs.end && step == rhs.step;
		}

		bool operator != (const step_range& rhs) const
		{
			return !(operator == (rhs));
		}

		class enumerator
		{
		public:
			typedef index_t value_type;

			enumerator(index_t b, index_t e, index_t s) : m_e(e), m_s(s), m_i(b-s) { }

			bool next()
			{
				m_i += m_s;
				return m_s < 0 ? m_i > m_e : m_i < m_e;
			}

			value_type get() const
			{
				return m_i;
			}

		private:
			index_t m_e;
			index_t m_s;
			index_t m_i;
		};

		enumerator get_enumerator() const
		{
			return enumerator(begin, end, step);
		}

	}; // end struct step_range



	struct whole
	{
		static const bool is_open_end = true;

		range close(size_t n) const
		{
			return range(0, (index_t)n);
		}

	}; // end struct whole


	struct open_range
	{
		static const bool is_open_end = true;

		index_t begin;
		index_t end_shift;

		open_range(index_t b, index_t es = 0) : begin(b), end_shift(es)
		{
		}

		range close(size_t n) const
		{
			return range(begin, (index_t)n + end_shift);
		}
	};


	struct open_step_range
	{
		static const bool is_open_end = true;

		index_t begin;
		index_t end_shift;
		index_t step;

		open_step_range(index_t b, index_t es, index_t s) : begin(b), end_shift(es), step(s)
		{
		}

		step_range close(size_t n) const
		{
			return step_range(begin, (index_t)n + end_shift, step);
		}
	};


	// Syntax sugar for constructing regular selector

	struct aend
	{
		index_t shift;

		explicit aend() : shift(0) { }
		explicit aend(index_t es) : shift(es) { }

		aend operator + (index_t s) const { return aend(shift + s); }
		aend operator - (index_t s) const { return aend(shift - s); }
	};


	inline range rgn(index_t begin, index_t end)
	{
		return range(begin, end);
	}

	inline step_range rgn(index_t begin, index_t end, index_t step)
	{
		return step_range(begin, end, step);
	}

	inline open_range rgn(index_t begin, aend end)
	{
		return open_range(begin, end.shift);
	}

	inline open_step_range rgn(index_t begin, aend end, index_t step)
	{
		return open_step_range(begin, end.shift, step);
	}


	// the selector based on a sequence of indices

	class indices
	{
	public:
		static const bool is_open_end = false;

		indices(const index_t *p, size_t n)
		: m_pblock(), m_pinds(p), m_n(n)
		{
		}

		indices(const std::vector<index_t>& refvec)
		: m_pblock(), m_pinds(&(refvec[0])), m_n((size_t)refvec.size())
		{
		}

		indices(const index_t *src, size_t n, clone_t)
		: m_pblock(new block<index_t>(n)), m_pinds(m_pblock->pbase()), m_n(n)
		{
			copy_elements(src, const_cast<index_t*>(m_pinds), n);
		}

		indices(const std::vector<index_t>& src, clone_t)
		: m_pblock(new block<index_t>(src.size())), m_pinds(m_pblock->pbase()), m_n(src.size())
		{
			copy_elements(&(src[0]), const_cast<index_t*>(m_pinds), src.size());
		}

		indices(block<index_t>* p_newblk, own_t)
		: m_pblock(p_newblk), m_pinds(p_newblk->pbase()), m_n(p_newblk->nelems())
		{
		}


		size_t size() const { return m_n; }

		bool is_empty() const { return m_n == 0; }

		bool is_contained_in(size_t n) const
		{
			for (index_t i = 0; i < (index_t)m_n; ++i)
			{
				index_t v = m_pinds[i];
				if (v < 0 || v >= (index_t)n) return false;
			}
			return true;
		}

		index_t operator[] (index_t i) const
		{
			return m_pinds[i];
		}

	public:

		class enumerator
		{
		public:
			typedef index_t value_type;

			enumerator(const indices& inds) : m_inds(inds), m_i(-1)
			{
			}

			bool next()
			{
				++ m_i;
				return m_i < (index_t)m_inds.size();
			}

			value_type get()
			{
				return m_inds[m_i];
			}
		private:
			const indices& m_inds;
			index_t m_i;
		};

		enumerator get_enumerator() const
		{
			return enumerator(*this);
		}


	private:
		tr1::shared_ptr<block<index_t> > m_pblock;
		const index_t *m_pinds;
		size_t m_n;


	}; // end class indices

}

#endif 
