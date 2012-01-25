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

	/***********************************************************
	 *
	 *   The Concept of Agent
	 *   ----------------------------------
	 *
	 *   Let a be an instance of an agent model
	 *
	 *	 a.source(v);
	 *	     invoked when a source vertex is to be added to
	 *	     the queue
	 *
	 *	     no return
	 *
	 *	 a.examine(u, v, v_status);
	 *	     invoked when an edge is examined.
	 *
	 *	     the agent should return whether the edge (u, v)
	 *	     is viable
	 *
	 *   a.discover(u, v);
	 *       invoked when a new vertex v is attained from u, which
	 *       is to be added to the queue
	 *
	 *       the agent should return whether to continue
	 *
	 *   a.finish(v);
	 *   	 invoked when all neighbors of v have been added
	 *   	 to the queue, and v is removed from the queue
	 *
	 *   	 the agent should return whether to continue
	 *
	 ***********************************************************/



	/***********************************************************
	 *
	 *   Breadth-First Traversal
	 *
	 ***********************************************************/

	enum gvisit_status
	{
		GVISIT_NONE = 0,
		GVISIT_DISCOVERED = 1,
		GVISIT_FINISHED = 2
	};


	template<class Derived, class Queue>
	class breadth_first_traverser
	{
	public:
		typedef Queue queue_type;
		typedef typename gview_traits<Derived>::vertex_type vertex_type;
		typedef typename gview_traits<Derived>::neighbor_iterator neighbor_iterator;

	public:
		breadth_first_traverser(const IGraphAdjacencyList<Derived>& g, Queue& queue)
		: m_graph(g), m_status(g.nvertices(), GVISIT_NONE), m_queue(queue)
		{
		}

		gvisit_status status(const vertex_type& v) const
		{
			return m_status[v];
		}

		template<class Agent>
		void add_source(const vertex_type& u, Agent& agent)
		{
			if (status(u) < GVISIT_DISCOVERED)
			{
				agent.source(u);
				m_status[u] = GVISIT_DISCOVERED;
				m_queue.push(u);
			}
		}


		template<class Agent> void run(Agent& agent);

	private:
		const IGraphAdjacencyList<Derived>& m_graph;
		array_map<vertex_type, gvisit_status> m_status;
		queue_type& m_queue;
	};


	template<class Derived, class Queue>
	template<class Agent>
	void breadth_first_traverser<Derived, Queue>::run(Agent& agent)
	{
		bool stopped = false;

		while (!m_queue.empty())
		{
			vertex_type u = m_queue.front();

			neighbor_iterator it = m_graph.out_neighbors_begin(u);
			neighbor_iterator nb_end = m_graph.out_neighbors_end(u);

			for (; it != nb_end; ++it)
			{
				vertex_type v = *it;
				gvisit_status vstat = status(v);

				if (agent.examine(u, v, vstat) && vstat < GVISIT_DISCOVERED)
				{
					m_status[v] = GVISIT_DISCOVERED;
					m_queue.push(v);

					if (!agent.discover(u, v))
					{
						stopped = true;
						break;
					}
				}
			}

			if (stopped) break;

			if (it == nb_end)
			{
				m_queue.pop();
				m_status[u] = GVISIT_FINISHED;
				if (!agent.finish(u)) break;
			}
		}
	}



	template<class Derived, class Agent>
	inline void breadth_first_traverse(const IGraphAdjacencyList<Derived>& g, Agent& agent,
			const typename gview_traits<Derived>::vertex_type& source)
	{
		typedef typename gview_traits<Derived>::vertex_type vertex_type;
		std::queue<vertex_type> Q;
		breadth_first_traverser<Derived, std::queue<vertex_type> > T(g, Q);

		T.add_source(source, agent);
		T.run(agent);
	}

	template<class Derived, class Agent, typename InputIter>
	inline void breadth_first_traverse(const IGraphAdjacencyList<Derived>& g, Agent& agent,
			InputIter first, InputIter last)
	{
		typedef typename gview_traits<Derived>::vertex_type vertex_type;
		std::queue<vertex_type> Q;
		breadth_first_traverser<Derived, std::queue<vertex_type> > T(g, Q);

		for (; first != last; ++first) T.add_source(*first, agent);
		T.run(agent);
	}



	/********************************************
	 *
	 *   Depth-First Traversal
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
			return m_status[v];
		}

		template<class Agent>
		void add_source(const vertex_type& u, Agent& agent)
		{
			if (status(u) < GVISIT_DISCOVERED)
			{
				agent.source(u);
				add_discovered(u);
			}
		}


		template<class Agent> void run(Agent& agent);

	private:

		void add_discovered(const vertex_type &v)
		{
			m_status[v] = GVISIT_DISCOVERED;

			entry e;
			e.v = v;
			e.nb_current = m_graph.out_neighbors_begin(v);
			e.nb_end = m_graph.out_neighbors_end(v);

			m_stack.push(e);
		}


	private:
		const IGraphAdjacencyList<Derived>& m_graph;
		array_map<vertex_type, gvisit_status> m_status;
		std::stack<entry> m_stack;
	};


	template<class Derived>
	template<class Agent>
	void depth_first_traverser<Derived>::run(Agent& agent)
	{
		while (!m_stack.empty())
		{
			entry& e = m_stack.top();

			if (e.nb_current != e.nb_end)
			{
				vertex_type v = *e.nb_current;
				++e.nb_current;
				gvisit_status vstatus = status(v);

				if (agent.examine(e.v, v, vstatus) && vstatus < GVISIT_DISCOVERED)
				{
					add_discovered(v);
					if (!agent.discover(e.v, v))
						break;
				}
			}
			else
			{
				vertex_type v = e.v;
				m_stack.pop();

				if (!agent.finish(v)) break;
			}
		}
	}


	template<class Derived, class Agent>
	inline void depth_first_traverse(const IGraphAdjacencyList<Derived>& g, Agent& agent,
			const typename gview_traits<Derived>::vertex_type& seed)
	{
		depth_first_traverser<Derived> T(g);

		T.add_source(seed, agent);
		T.run(agent);
	}

	template<class Derived, class Agent, typename InputIter>
	inline void depth_first_traverse(const IGraphAdjacencyList<Derived>& g, Agent& agent,
			InputIter first, InputIter last)
	{
		depth_first_traverser<Derived> T(g);

		for (; first != last; ++first)
		{
			T.add_source(*first, agent);
			T.run(agent);
		}
	}

}


#endif /* GRAPH_TRAVERSAL_H_ */
