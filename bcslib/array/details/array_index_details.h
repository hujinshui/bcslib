/**
 * @file array_index_details.cpp
 *
 * Details for array_index.cpp
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY_INDEX_DETAILS_CPP_
#define BCSLIB_ARRAY_INDEX_DETAILS_CPP_

namespace bcs
{
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
}

#endif
