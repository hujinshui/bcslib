/**
 * @file disjoint_sets.h
 *
 *  The data structure for representing disjoint sets
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#include <bcslib/base/basic_defs.h>
#include <vector>

#ifndef BCSLIB_DISJOINT_SETS_H_
#define BCSLIB_DISJOINT_SETS_H_

namespace bcs
{

	class disjoint_set_forest
	{
	public:
		typedef size_t size_type;

		struct node
		{
			size_type rank;
			size_type parent;

			node(size_type i)
			{
				rank = 0;
				parent = i;
			}
		};

	public:
		disjoint_set_forest(size_type n)
		{
			m_nodes.reserve(n);
			for (size_type i = 0; i < n; ++i)
			{
				m_nodes.push_back(node(i));
			}
			m_ncomps = n;
		}

		size_type size() const
		{
			return m_nodes.size();
		}

		size_type ncomponents() const
		{
			return m_ncomps;
		}

		bool is_root(const size_type& x) const
		{
			return x == parent(x);
		}

		size_type parent(const size_type& x) const
		{
			return m_nodes[x].parent;
		}

		size_type rank(const size_type& x) const
		{
			return m_nodes[x].rank;
		}

		bool in_same_component(const size_type& x, const size_type& y)
		{
			return find_root(x) == find_root(y);
		}

	public:

		/**
		 * Returns true if x and y were not in the same set, and thus
		 * the joining was actually performed
		 */
		bool join(const size_type& x, const size_type& y)
		{
			size_type rx = find_root(x);
			size_type ry = find_root(y);

			if (rx != ry)
			{
				size_type r = link_root(rx, ry);

				m_nodes[x].parent = r;
				m_nodes[y].parent = r;

				return true;
			}
			else
			{
				return false;
			}
		}

		size_type find_root(const size_type& x)
		{
			node& nx = m_nodes[x];
			if (x != nx.parent)
			{
				return nx.parent = find_root(nx.parent);
			}
			else
			{
				return x;
			}
		}

		void compress_all()
		{
			size_type n = size();
			for (size_type i = 0; i < n; ++i)
			{
				size_type ri = find_root(i);
				m_nodes[i].rank = (ri == i ? 1 : 0);
			}
		}

		size_type trace_root(const size_type& x) const
		{
			size_type p = x;
			size_type pp = parent(p);

			while (p != pp)
			{
				p = pp;
				pp = parent(p);
			}

			return p;
		}

	private:
		size_type link_root(const size_type& rx, const size_type & ry)
		{
			node& nx = m_nodes[rx];
			node& ny = m_nodes[ry];

			-- m_ncomps;

			if (nx.rank > ny.rank)
			{
				ny.parent = rx;

				return rx;
			}
			else
			{
				nx.parent = ry;
				if (nx.rank == ny.rank) ++ ny.rank;

				return ry;
			}
		}

	private:
		std::vector<node> m_nodes;
		size_type m_ncomps;

	}; // end class disjoint_set_forest

}

#endif /* DISJOINT_SET_H_ */
