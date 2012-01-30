/**
 * @file static_tree.h
 *
 * The class to represent a static tree structure
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_STATIC_TREE_H_
#define BCSLIB_STATIC_TREE_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/key_map.h>
#include <bcslib/base/iterator_wrappers.h>
#include <vector>

namespace bcs
{

	template<typename T>
	class static_tree
	{
	public:
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(index_convertible<T>::value, "TKey must be index-convertible");
#endif
		typedef T element_type;
		typedef size_t size_type;
		typedef index_t index_type;

		struct node
		{
			element_type e;

			index_type parent_index;
			std::vector<index_type> children_indices;
		};

	private:
		class children_iter_impl
		{
		public:
			typedef T value_type;
			typedef const T& reference;
			typedef const T* pointer;

		public:
			children_iter_impl(const T *nodes,
					typename std::vector<index_type>::const_iterator pind)
			: m_nodes(nodes), m_pind(pind)
			{
			}

			bool operator == (const children_iter_impl& rhs) const
			{
				return m_pind == rhs.m_pind;
			}

			void move_next()
			{
				++ m_pind;
			}

			pointer ptr() const
			{
				return &(ref());
			}

			reference ref() const
			{
				return m_nodes[*m_pind].e;
			}

		private:
			const node *m_nodes;
			const typename std::vector<index_type>::const_iterator m_pind;
		};

	public:
		typedef forward_iterator_wrapper<children_iter_impl> children_iterator;


	public:
		BCS_ENSURE_INLINE
		size_type size() const
		{
			return m_nodes.size();
		}

		BCS_ENSURE_INLINE
		const T& get(const index_type& idx) const
		{
			return m_nodes[idx].e;
		}

		BCS_ENSURE_INLINE
		index_type root_index() const
		{
			return m_root_index;
		}

		BCS_ENSURE_INLINE
		const T& root() const
		{
			return m_nodes[m_root_index].e;
		}

		BCS_ENSURE_INLINE
		bool is_root(const T& e) const
		{
			return index(e) == m_root_index;
		}

		BCS_ENSURE_INLINE
		bool is_leaf(const T& e) const
		{
			return get_node(e).children_indices.empty();
		}

		BCS_ENSURE_INLINE
		const T& parent(const T& e) const
		{
			return get(get_node(e).parent_index);
		}

		BCS_ENSURE_INLINE
		bool has_children(const T& e) const
		{
			return !is_leaf(e);
		}

		BCS_ENSURE_INLINE
		size_type nchildren(const T& e) const
		{
			return get_node(e).children_indices.size();
		}

		BCS_ENSURE_INLINE
		const T& child(const T& e, const size_type& i) const
		{
			return get(get_node(e).children_indices[i]);
		}

		BCS_ENSURE_INLINE
		children_iterator children_begin(const T& e) const
		{
			const node& nd = get_node(e);
			return children_iter_impl(&(m_nodes.front()),
					nd.children_indices.begin());
		}

		BCS_ENSURE_INLINE
		children_iterator children_end(const T& e) const
		{
			const node& nd = get_node(e);
			return children_iter_impl(&(m_nodes.front()),
					nd.children_indices.end());
		}

	public:
		template<class InputIterator, class ParentIterator>
		void create_from_parents(InputIterator first, InputIterator last, ParentIterator parents);

	private:
		BCS_ENSURE_INLINE
		index_type index(const T& e) const
		{
			return key_to_index<T>::to_index(e);
		}

		BCS_ENSURE_INLINE
		const node& get_node(const T& e) const
		{
			return m_nodes[index(e)];
		}

	private:
		index_type m_root_index;
		std::vector<node> m_nodes;

	}; // end class static_tree


	template<typename T>
	template<class InputIterator, class ParentIterator>
	void static_tree<T>::create_from_parents(InputIterator first, InputIterator last, ParentIterator parents)
	{

	}



}


#endif /* STATIC_TREE_H_ */
