/**
 * @file bgl_port.h
 *
 * Adapting the graph classes to BGL concepts
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_BGL_PORT_H
#define BCSLIB_BGL_PORT_H

#include <bcslib/graph/graph_base.h>
#include <bcslib/graph/gr_edgelist.h>
#include <bcslib/graph/gr_adjlist.h>

#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_concepts.hpp>

namespace boost
{
	template<typename TDir> struct bcs_bgl_dir_translator;

	template<>
	struct bcs_bgl_dir_translator<bcs::gr_directed>
	{
		typedef boost::directed_tag directed_category;
	};


	template<typename TDir>
	struct graph_traits<bcs::gr_edgelist<TDir> >
	{
		typedef bcs::gr_edgelist<TDir> graph_type;

		typedef typename graph_type::vertex_type vertex_descriptor;
		typedef typename graph_type::edge_type edge_descriptor;

		typedef bcs::gr_size_t vertices_size_type;
		typedef bcs::gr_size_t edges_size_type;

		struct traversal_category :
			public virtual vertex_list_graph_tag,
			public virtual edge_list_graph_tag { };

		typedef typename bcs_bgl_dir_translator<TDir>::directed_category directed_category;
		typedef disallow_parallel_edge_tag edge_parallel_category;

		typedef typename graph_type::vertex_iterator vertex_iterator;
		typedef typename graph_type::edge_iterator edge_iterator;
	};


	template<typename TWeight, typename TDir>
	struct graph_traits<bcs::gr_wedgelist<TWeight, TDir> >
	{
		typedef bcs::gr_edgelist<TDir> graph_type;

		typedef typename graph_type::vertex_type vertex_descriptor;
		typedef typename graph_type::edge_type edge_descriptor;

		typedef bcs::gr_size_t vertices_size_type;
		typedef bcs::gr_size_t edges_size_type;

		struct traversal_category :
			public virtual vertex_list_graph_tag,
			public virtual edge_list_graph_tag { };

		typedef typename bcs_bgl_dir_translator<TDir>::directed_category directed_category;
		typedef disallow_parallel_edge_tag edge_parallel_category;

		typedef typename graph_type::vertex_iterator vertex_iterator;
		typedef typename graph_type::edge_iterator edge_iterator;
	};


	template<typename TDir>
	struct graph_traits<bcs::gr_adjlist<TDir> >
	{
		typedef bcs::gr_adjlist<TDir> graph_type;

		typedef typename graph_type::vertex_type vertex_descriptor;
		typedef typename graph_type::edge_type edge_descriptor;

		typedef bcs::gr_size_t vertices_size_type;
		typedef bcs::gr_size_t edges_size_type;
		typedef bcs::gr_size_t degree_size_type;

		struct traversal_category :
			public virtual vertex_list_graph_tag,
			public virtual edge_list_graph_tag,
			public virtual incidence_graph_tag,
			public virtual adjacency_graph_tag { };

		typedef typename bcs_bgl_dir_translator<TDir>::directed_category directed_category;
		typedef disallow_parallel_edge_tag edge_parallel_category;

		typedef typename graph_type::vertex_iterator vertex_iterator;
		typedef typename graph_type::edge_iterator edge_iterator;
		typedef typename graph_type::adj_edge_iterator out_edge_iterator;
		typedef typename graph_type::neighbor_iterator adjacency_iterator;
	};


	template<typename TWeight, typename TDir>
	struct graph_traits<bcs::gr_wadjlist<TWeight, TDir> >
	{
		typedef bcs::gr_wadjlist<TWeight, TDir> graph_type;

		typedef typename graph_type::vertex_type vertex_descriptor;
		typedef typename graph_type::edge_type edge_descriptor;

		typedef bcs::gr_size_t vertices_size_type;
		typedef bcs::gr_size_t edges_size_type;
		typedef bcs::gr_size_t degree_size_type;

		struct traversal_category :
			public virtual vertex_list_graph_tag,
			public virtual edge_list_graph_tag,
			public virtual incidence_graph_tag,
			public virtual adjacency_graph_tag { };

		typedef typename bcs_bgl_dir_translator<TDir>::directed_category directed_category;
		typedef disallow_parallel_edge_tag edge_parallel_category;

		typedef typename graph_type::vertex_iterator vertex_iterator;
		typedef typename graph_type::edge_iterator edge_iterator;
		typedef typename graph_type::adj_edge_iterator out_edge_iterator;
		typedef typename graph_type::neighbor_iterator adjacency_iterator;
	};

}


#endif 
