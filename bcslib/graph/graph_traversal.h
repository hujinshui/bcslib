/**
 * @file graph_traversal.h
 *
 * Graph algorithms for graph traversal
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef GRAPH_TRAVERSAL_H_
#define GRAPH_TRAVERSAL_H_

#include <bcslib/graph/gview_base.h>
#include <bcslib/array/amap.h>
#include <vector>
#include <stack>
#include <queue>
#include <utility>

namespace bcs
{

	/********************************************
	 *
	 *   Breadth-First Traversal
	 *
	 *
	 *   Visitor concept
	 *   ---------------
	 *
	 *	 vis.seed(v);
	 *	 vis.examine(u, v, v_status);
	 *   vis.discover(u, v);
	 *   vis.finish(v);
	 *
	 ********************************************/

	enum gvisit_status
	{
		GVISIT_NONE = 0,
		GVISIT_DISCOVERED = 1,
		GVISIT_FINISHED = 2
	};


	template<class Derived>
	class breadth_first_traverser
	{
	public:
		typedef typename gview_traits<Derived>::vertex_type vertex_type;
		typedef typename gview_traits<Derived>::neighbor_iterator neighbor_iterator;

	public:
		breadth_first_traverser(const IGraphAdjacencyList<Derived>& g)
		: m_graph(g), m_status(g.nvertices(), GVISIT_NONE)
		{
		}

		gvisit_status status(const vertex_type& v) const
		{
			return m_status(v);
		}

		bool is_ended() const
		{
			return m_queue.empty();
		}

		template<class Visitor>
		void add_seed(const vertex_type& v, Visitor& visitor)
		{
			if (status(v) < GVISIT_DISCOVERED)
			{
				visitor.seed(v);
				m_status(v) = GVISIT_DISCOVERED;
				add_neighbors(v, visitor);
			}
		}


		template<class Visitor>
		void next(Visitor& visitor)
		{
			vertex_type v = m_queue.front();
			m_queue.pop();

			add_neighbors(v, visitor);
		}

	private:
		template<class Visitor>
		void add_neighbors(const vertex_type& v, Visitor& visitor)
		{
			neighbor_iterator nb_begin = m_graph.out_neighbors_begin(v);
			neighbor_iterator nb_end = m_graph.out_neighbors_end(v);

			for (neighbor_iterator it = nb_begin; it != nb_end; ++it)
			{
				vertex_type u = *it;

				gvisit_status ustat = status(u);
				visitor.examine(v, u, ustat);

				if (status(u) < GVISIT_DISCOVERED)
				{
					m_status(u) = GVISIT_DISCOVERED;
					visitor.discover(v, u);
					m_queue.push(u);
				}
			}

			m_status(v) = GVISIT_FINISHED;
			visitor.finish(v);
		}

	private:
		const IGraphAdjacencyList<Derived>& m_graph;
		array_map<vertex_type, gvisit_status> m_status;
		std::queue<vertex_type> m_queue;
	};


	template<class Derived, class Visitor>
	inline void breadth_first_traverse(const IGraphAdjacencyList<Derived>& g, Visitor& visitor,
			const typename gview_traits<Derived>::vertex_type& seed)
	{
		breadth_first_traverser<Derived> T(g);

		T.add_seed(seed, visitor);

		while (!T.is_ended())
		{
			T.next(visitor);
		}
	}

	template<class Derived, class Visitor, typename InputIter>
	inline void breadth_first_traverse(const IGraphAdjacencyList<Derived>& g, Visitor& visitor,
			InputIter first, InputIter last)
	{
		breadth_first_traverser<Derived> T(g);

		for (; first != last; ++first) T.add_seed(*first, visitor);

		while (!T.is_ended())
		{
			T.next(visitor);
		}
	}



	/********************************************
	 *
	 *   Depth-First Traversal
	 *
	 *
	 *   Visitor concept
	 *   ---------------
	 *
	 *   vis.seed(v);
	 *   vis.examine(u, v, v_status);
	 *   vis.discover(u, v);
	 *   vis.finish(v);
	 *
	 ********************************************/

	template<class Derived>
	class depth_first_traverser
	{
	public:
		typedef typename gview_traits<Derived>::vertex_type vertex_type;
		typedef typename gview_traits<Derived>::neighbor_iterator neighbor_iterator;

		struct entry
		{
			vertex_type v;
			neighbor_iterator nb_current;
			neighbor_iterator nb_end;
		};

	public:
		depth_first_traverser(const IGraphAdjacencyList<Derived>& g)
		: m_graph(g), m_status(g.nvertices(), GVISIT_NONE)
		{

		}

		gvisit_status status(const vertex_type& v) const
		{
			return m_status(v);
		}

		bool is_ended() const
		{
			return m_stack.empty();
		}

		template<class Visitor>
		void add_seed(const vertex_type& v, Visitor& visitor)
		{
			if (status(v) < GVISIT_DISCOVERED)
			{
				visitor.seed(v);
				add_discovered(v, visitor);
			}
		}

		template<class Visitor>
		void next(Visitor& visitor)
		{
			entry& e = m_stack.top();

			if (e.nb_current != e.nb_end)
			{
				vertex_type v = *e.nb_current;

				gvisit_status vstatus = status(v);
				visitor.examine(e.v, v, vstatus);

				bool pushed = false;

				if (vstatus < GVISIT_DISCOVERED)
				{
					visitor.discover(e.v, v);
					pushed = add_discovered(v, visitor);
				}

				if (!pushed) ++e.nb_current;
			}
			else
			{
				m_status(e.v) = GVISIT_FINISHED;
				visitor.finish(e.v);
				m_stack.pop();

				if (!m_stack.empty())
				{
					++ m_stack.top().nb_current;
				}
			}

		}

	private:

		template<class Visitor>
		bool add_discovered(const vertex_type &v, Visitor& visitor)
		{
			m_status(v) = GVISIT_DISCOVERED;

			entry e;
			e.v = v;
			e.nb_current = m_graph.out_neighbors_begin(v);
			e.nb_end = m_graph.out_neighbors_end(v);

			if (e.nb_current != e.nb_end)
			{
				m_stack.push(e);
				return true;
			}
			else
			{
				m_status(v) = GVISIT_FINISHED;
				visitor.finish(v);
				return false;
			}
		}

	private:
		const IGraphAdjacencyList<Derived>& m_graph;
		array_map<vertex_type, gvisit_status> m_status;
		std::stack<entry> m_stack;
	};


	template<class Derived, class Visitor>
	inline void depth_first_traverse(const IGraphAdjacencyList<Derived>& g, Visitor& visitor,
			const typename gview_traits<Derived>::vertex_type& seed)
	{
		depth_first_traverser<Derived> T(g);

		T.add_seed(seed, visitor);

		while (!T.is_ended())
		{
			T.next(visitor);
		}
	}

	template<class Derived, class Visitor, typename InputIter>
	inline void depth_first_traverse(const IGraphAdjacencyList<Derived>& g, Visitor& visitor,
			InputIter first, InputIter last)
	{
		depth_first_traverser<Derived> T(g);

		for (; first != last; ++first)
		{
			T.add_seed(*first, visitor);

			while (!T.is_ended())
			{
				T.next(visitor);
			}
		}
	}

}


#endif /* GRAPH_TRAVERSAL_H_ */
