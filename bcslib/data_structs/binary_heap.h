/**
 * @file binary_heap.h
 *
 * The class that implements a binary heap
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BINARY_HEAP_H_
#define BINARY_HEAP_H_

#include <bcslib/base/basic_defs.h>
#include <vector>
#include <functional>


namespace bcs
{

	template<typename T,
		class Container=std::vector<T>,
		class Compare=std::less<T> >
	class binary_heap
	{
	public:
		typedef Container container_type;
		typedef Compare compare_type;

		typedef T value_type;
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		typedef typename container_type::reference reference;
		typedef typename container_type::const_reference const_reference;
		typedef typename container_type::pointer pointer;
		typedef typename container_type::const_pointer const_pointer;
		typedef typename container_type::iterator iterator;
		typedef typename container_type::const_iterator const_iterator;
		typedef typename container_type::allocator_type allocator_type;

		typedef size_t node_type;

	public:
		binary_heap(container_type& elements)
		: m_elements(elements)
		{
			init_heap();
		}

		binary_heap(container_type& elements, size_type initcap)
		: m_elements(elements)
		{
			init_heap();
			m_btree.reserve(initcap);
		}

	public:
		size_type size() const
		{
			return m_btree.size();
		}

		bool empty() const
		{
			return m_btree.empty();
		}

		const_reference top() const
		{
			return m_elements[m_btree[0]];
		}

		const_reference get(size_type i) const
		{
			return m_elements[i];
		}

		void set(size_type i, const value_type& e)
		{
			const_reference e0 = m_elements[i];
			if (m_compare(e, e0))
			{
				m_elements[i] = e;
				bubble_up(m_node_map[i], e);
			}
			else
			{
				m_elements[i] = e;
				bubble_down(m_node_map[i], e);
			}
		}

		bool is_in_heap(size_type i)
		{
			return m_node_map[i] > 0;
		}

	public:

		// for inspection & debug

		const std::vector<size_type>& tree_array() const
		{
			return m_btree;
		}

		const std::vector<node_type>& node_map() const
		{
			return m_node_map;
		}

		const container_type& elements() const
		{
			return m_elements;
		}

		const_reference get_by_node(node_type u) const
		{
			return m_elements[m_btree[u - 1]];
		}

	public:
		void push(const T& e)
		{
			m_elements.push_back(e);

			size_type i = m_elements.size() - 1;
			m_btree.push_back(i);

			node_type last_node = m_btree.size();
			m_node_map.push_back(last_node);

			if (last_node > 1)
			{
				bubble_up(last_node, e);
			}
		}

		void pop()
		{
			size_type n = m_btree.size();

			if (n > 0)
			{
				m_node_map[m_btree[0]] = 0;

				if (n > 1)
				{
					size_type i = m_btree.back();
					m_btree[0] = i;

					node_type rt = root_node();
					m_node_map[i] = rt;

					bubble_down(rt, m_elements[i]);
				}

				m_btree.pop_back();
			}
		}

	private:

		void init_heap()
		{
			size_type n = m_elements.size();
			for (size_type i = 0; i < n; ++i)
			{
				m_btree.push_back(i);
				m_node_map.push_back(last_node());
			}

			if (n > 1)
			{
				node_type last_nonleaf = parent(last_node());
				for (size_type v = last_nonleaf; v > 0; --v)
				{
					bubble_down(v, m_elements[v-1]);
				}
			}
		}

		void bubble_up(node_type u, const value_type& e)
		{
			bool cont = true;

			while (cont && u > root_node())
			{
				node_type p = parent(u);
				if (m_compare(e, get_by_node(p)))
				{
					u = swap(u, p);
				}
				else
				{
					cont = false;
				}
			}
		}

		void bubble_down(node_type u, const value_type& e)
		{
			bool cont = true;

			node_type last_nl = parent(last_node());
			while (cont && u <= last_nl)
			{
				cont = false;

				node_type lc, rc;
				get_children(u, lc, rc);

				if (rc > 0)
				{
					const_reference le = get_by_node(lc);
					const_reference re = get_by_node(rc);

					if (m_compare(le, re))
					{
						if (m_compare(le, e))
						{
							u = swap(u, lc);
							cont = true;
						}
					}
					else
					{
						if (m_compare(re, e))
						{
							u = swap(u, rc);
							cont = true;
						}
					}
				}
				else if (lc > 0)
				{
					const_reference le = get_by_node(lc);
					if (m_compare(le, e))
					{
						u = swap(u, lc);
						cont = true;
					}
				}
			}
		}

		node_type swap(node_type u, node_type v)
		{
			size_type ui = m_btree[u - 1];
			size_type vi = m_btree[v - 1];

			m_node_map[ui] = v;
			m_node_map[vi] = u;

			m_btree[u - 1] = vi;
			m_btree[v - 1] = ui;

			return v;
		}

	private:
		node_type last_node() const
		{
			return m_btree.size();
		}

		node_type root_node() const
		{
			return 1;
		}

		node_type parent(node_type v) const
		{
			return v >> 1;
		}

		void get_children(node_type v, node_type& lc, node_type &rc) const
		{
			lc = v << 1;

			node_type last = last_node();

			if (lc < last)
			{
				rc = lc + 1;
			}
			else if (lc == last)
			{
				rc = 0;
			}
			else
			{
				lc = 0;
				rc = 0;
			}
		}

	private:
		container_type& m_elements;
		std::vector<size_type> m_btree;
		std::vector<node_type> m_node_map;

		Compare m_compare;

	}; // end class binary_heap


}

#endif /* STATIC_BINARY_TREE_H_ */
