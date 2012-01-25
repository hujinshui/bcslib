/*
 * @file graph_shortest_paths.h
 *
 * Shortest path algorithms
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif


#ifndef BCSLIB_GRAPH_SHORTEST_PATHS_H_
#define BCSLIB_GRAPH_SHORTEST_PATHS_H_

#include <bcslib/graph/gview_base.h>
#include <bcslib/graph/graph_traversal.h>
#include <bcslib/data_structs/binary_heap.h>


namespace bcs
{

	template<class Derived, typename TDist, class EdgeDistMap, class PathLenMap, class Heap>
	class dijkstra_traverser
	{
	public:
		typedef gview_traits<Derived>& vertex_type;
		typedef TDist distance_type;
		typedef EdgeDistMap edge_distance_map_type;
		typedef PathLenMap path_length_map_type;
		typedef Heap heap_type;

		typedef typename gview_traits<Derived>::vertex_iterator vertex_iterator;
		typedef typename gview_traits<Derived>::incident_edge_iterator incident_edge_iterator;
		typedef typename heap_type::node_type heap_node_type;

	public:
		dijkstra_traverser(const IGraphIncidenceList<Derived>& g, const TDist& initlen)
		: m_graph(g)
		, m_status(g.nvertices(), GVISIT_NONE)
		, m_heap_node_map(g.nvertices(), heap_node_type())
		, m_heap(m_shortest_path_lens, m_heap_node_map)
		{
			vertex_iterator pv = g.vertices_begin();
			vertex_iterator vend = g.vertices_end();

			for (; pv != vend; ++pv)
			{
				const vertex_type& v = *pv;

				m_status[v] = GVISIT_NONE;
				m_shortest_path_lens[v] = initlen;
			}
		}

		void add_source(const vertex_type& v)
		{

		}

		void next()
		{

		}

	private:
		void relax(const vertex_type& u, const vertex_type& v, const distance_type& new_dist)
		{

		}

	private:
		const IGraphIncidenceList<Derived>& m_graph;
		array_map<vertex_type, gvisit_status> m_status;
		const edge_distance_map_type& m_edge_dists;
		path_length_map_type& m_shortest_path_lens;

		array_map<vertex_type, heap_node_type> m_heap_node_map;
		heap_type m_heap;

	}; // end class dijkstra_traverser



}

#endif /* GRAPH_SHORTEST_PATHS_H_ */
