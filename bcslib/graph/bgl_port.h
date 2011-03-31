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


namespace bcs
{

	/************************************************
	 *
	 *  Property maps
	 *
	 ***********************************************/

	// vertex -> index

	class vertex_index_map
	{
	public:
		typedef vertex_t key_type;
		typedef gr_index_t value_type;
		typedef gr_index_t reference;
		typedef boost::readable_property_map_tag category;

	public:
		reference get(const key_type& v) const
		{
			return v.index;
		}
	};

	inline gr_index_t get(const vertex_index_map& m, const vertex_t& v)
	{
		return m.get(v);
	}

	template<class TGraph>
	inline vertex_index_map get(boost::vertex_index_t, const TGraph& g)
	{
		return vertex_index_map();
	}

	template<class TGraph>
	inline gr_index_t get(boost::vertex_index_t, const TGraph& g, const vertex_t &v)
	{
		return v.index;
	}


	// edge -> index

	class edge_index_map
	{
	public:
		typedef edge_t key_type;
		typedef gr_index_t value_type;
		typedef gr_index_t reference;
		typedef boost::readable_property_map_tag category;

	public:
		reference get(const key_type& e) const
		{
			return e.index;
		}
	};

	inline gr_index_t get(const edge_index_map& m, const edge_t& e)
	{
		return m.get(e);
	}

	template<typename TGraph>
	inline edge_index_map get(boost::edge_index_t, const TGraph& g)
	{
		return edge_index_map();
	}

	template<typename TGraph>
	inline gr_index_t get(boost::edge_index_t, const TGraph& g, const edge_t &e)
	{
		return e.index;
	}


	// vertex -> value


	template<typename T>
	class vertex_cref_map
	{
	public:
		typedef vertex_t key_type;
		typedef T value_type;
		typedef const value_type& reference;
		typedef boost::readable_property_map_tag category;

		vertex_cref_map(const T *vals) : m_values(const_cast<T*>(vals))
		{
		}

		const value_type& operator[] (const vertex_t& v) const
		{
			return m_values[v.index];
		}

	protected:
		T *m_values;
	};

	template<typename T>
	class vertex_ref_map : public vertex_cref_map<T>
	{
	public:
		typedef vertex_t key_type;
		typedef T value_type;
		typedef value_type& reference;
		typedef boost::lvalue_property_map_tag category;

		vertex_ref_map(T *vals) : vertex_cref_map<T>(vals)
		{
		}

		const value_type& operator[] (const vertex_t& v) const
		{
			return this->m_values[v.index];
		}

		value_type& operator[] (const vertex_t& v)
		{
			return this->m_values[v.index];
		}
	};

	template<typename T>
	const T& get(const vertex_cref_map<T>& m, const vertex_t& v)
	{
		return m[v.index];
	}

	template<typename T>
	const T& get(const vertex_ref_map<T>& m, const vertex_t& v)
	{
		return m[v.index];
	}

	template<typename T>
	void put(vertex_ref_map<T>& m, const vertex_t& v, const T& c)
	{
		m[v.index] = c;
	}


	// edge -> value

	template<typename T>
	class edge_cref_map
	{
	public:
		typedef edge_t key_type;
		typedef T value_type;
		typedef const value_type& reference;
		typedef boost::readable_property_map_tag category;

		edge_cref_map(const T *vals) : m_values(const_cast<T*>(vals))
		{
		}

		const value_type& operator[] (const edge_t& e) const
		{
			return m_values[e.index];
		}

	protected:
		T *m_values;
	};


	template<typename T>
	class edge_ref_map : public edge_cref_map<T>
	{
	public:
		typedef edge_t key_type;
		typedef T value_type;
		typedef value_type& reference;
		typedef boost::lvalue_property_map_tag category;

		edge_ref_map(T *vals) : edge_cref_map<T>(vals)
		{
		}

		const value_type& operator[] (const edge_t& e) const
		{
			return this->m_values[e.index];
		}

		value_type& operator[] (const edge_t& e)
		{
			return this->m_values[e.index];
		}
	};


	template<typename T>
	const T& get(const edge_cref_map<T>& m, const edge_t& e)
	{
		return m[e.index];
	}

	template<typename T>
	const T& get(const edge_ref_map<T>& m, const edge_t& e)
	{
		return m[e.index];
	}

	template<typename T>
	void put(edge_ref_map<T>& m, const edge_t& e, const T& c)
	{
		m[e.index] = c;
	}


	/************************************************
	 *
	 *  Adapting functions for specific graph classes
	 *
	 ***********************************************/

	// for gr_edgelist (and thus gr_wedgelist)

	template<typename TDir>
	inline gr_size_t num_vertices(const gr_edgelist<TDir>& g)
	{
		return g.nvertices();
	}


	template<typename TDir>
	inline std::pair<simple_vertex_iterator, simple_vertex_iterator> vertices(const gr_edgelist<TDir>& g)
	{
		return std::make_pair(g.v_begin(), g.v_end());
	}


	template<typename TDir>
	inline gr_size_t num_edges(const gr_edgelist<TDir>& g)
	{
		return g.nedges();
	}


	template<typename TDir>
	inline std::pair<simple_edge_iterator, simple_edge_iterator> edges(const gr_edgelist<TDir>& g)
	{
		return std::make_pair(g.e_begin(), g.e_end());
	}


	template<typename TDir>
	inline vertex_index_map get(boost::vertex_index_t, const gr_edgelist<TDir>& g)
	{
		return vertex_index_map();
	}

	template<typename TDir>
	inline edge_index_map get(boost::edge_index_t, const gr_edgelist<TDir>& g)
	{
		return edge_index_map();
	}

	template<typename TWeight, typename TDir>
	inline edge_cref_map<TWeight> get(boost::edge_weight_t, const gr_wedgelist<TWeight, TDir>& g)
	{
		return g.weights();
	}


	// for gr_adjlist (and thus gr_wadjlist)


	template<typename TDir>
	inline gr_size_t num_vertices(const gr_adjlist<TDir>& g)
	{
		return g.nvertices();
	}


	template<typename TDir>
	inline std::pair<simple_vertex_iterator, simple_vertex_iterator> vertices(const gr_adjlist<TDir>& g)
	{
		return std::make_pair(g.v_begin(), g.v_end());
	}


	template<typename TDir>
	inline gr_size_t num_edges(const gr_adjlist<TDir>& g)
	{
		return g.nedges();
	}


	template<typename TDir>
	inline std::pair<simple_edge_iterator, simple_edge_iterator> edges(const gr_adjlist<TDir>& g)
	{
		return std::make_pair(g.e_begin(), g.e_end());
	}

	template<typename TDir>
	inline gr_size_t out_degree(const vertex_t& v, const gr_adjlist<TDir>& g)
	{
		return g.out_degree(v);
	}

	template<typename TDir>
	inline std::pair<const edge_t*, const edge_t*> out_edges(const vertex_t& v, const gr_adjlist<TDir>& g)
	{
		return std::make_pair(g.out_edge_begin(v), g.out_edge_end(v));
	}


	template<typename TDir>
	inline std::pair<const vertex_t*, const vertex_t*> adjacency_vertices(const vertex_t& v, const gr_adjlist<TDir>& g)
	{
		return std::make_pair(g.out_neighbor_begin(v), g.out_neighbor_end(v));
	}

	template<typename TDir>
	inline vertex_index_map get(boost::vertex_index_t, const gr_adjlist<TDir>& g)
	{
		return vertex_index_map();
	}

	template<typename TDir>
	inline edge_index_map get(boost::edge_index_t, const gr_adjlist<TDir>& g)
	{
		return edge_index_map();
	}

	template<typename TWeight, typename TDir>
	inline edge_cref_map<TWeight> get(boost::edge_weight_t, const gr_wadjlist<TWeight, TDir>& g)
	{
		return g.weights();
	}





}  // end namespace bcs




namespace boost
{
	/************************************************
	 *
	 *  graph_traits and property map traits
	 *
	 ***********************************************/

	template<typename TDir> struct bcs_bgl_dir_translator;

	template<>
	struct bcs_bgl_dir_translator<bcs::gr_directed>
	{
		typedef boost::directed_tag directed_category;
	};


	// for gr_edgelist

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

	template<typename TDir>
	struct property_map<bcs::gr_edgelist<TDir>, boost::vertex_index_t>
	{
		typedef bcs::vertex_index_map const_type;
		typedef bcs::vertex_index_map type;
	};

	template<typename TDir>
	struct property_map<bcs::gr_edgelist<TDir>, boost::edge_index_t>
	{
		typedef bcs::edge_index_map const_type;
		typedef bcs::edge_index_map type;
	};


	// for gr_wedgelist

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

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wedgelist<TWeight, TDir>, boost::vertex_index_t>
	{
		typedef bcs::vertex_index_map const_type;
		typedef bcs::vertex_index_map type;
	};

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wedgelist<TWeight, TDir>, boost::edge_index_t>
	{
		typedef bcs::edge_index_map const_type;
		typedef bcs::edge_index_map type;
	};

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wedgelist<TWeight, TDir>, boost::edge_weight_t>
	{
		typedef bcs::edge_cref_map<TWeight> const_type;
		typedef bcs::edge_ref_map<TWeight> type;
	};


	// for gr_adjlist

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

	template<typename TDir>
	struct property_map<bcs::gr_adjlist<TDir>, boost::vertex_index_t>
	{
		typedef bcs::vertex_index_map const_type;
		typedef bcs::vertex_index_map type;
	};

	template<typename TDir>
	struct property_map<bcs::gr_adjlist<TDir>, boost::edge_index_t>
	{
		typedef bcs::edge_index_map const_type;
		typedef bcs::edge_index_map type;
	};


	// for gr_wadjlist

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

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wadjlist<TWeight, TDir>, boost::vertex_index_t>
	{
		typedef bcs::vertex_index_map const_type;
		typedef bcs::vertex_index_map type;
	};

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wadjlist<TWeight, TDir>, boost::edge_index_t>
	{
		typedef bcs::edge_index_map const_type;
		typedef bcs::edge_index_map type;
	};

	template<typename TWeight, typename TDir>
	struct property_map<bcs::gr_wadjlist<TWeight, TDir>, boost::edge_weight_t>
	{
		typedef bcs::edge_cref_map<TWeight> const_type;
		typedef bcs::edge_ref_map<TWeight> type;
	};


} // end namespace boost




#endif 
