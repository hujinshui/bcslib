/*
 * @file gr_edgelist.h
 *
 * The class to represent a graph as edge list
 *
 *  @author dhlin
 */

#ifndef BCSLIB_GR_EDGELIST_H
#define BCSLIB_GR_EDGELIST_H

#include <bcslib/graph/graph_base.h>


namespace bcs
{

	template<typename TDir>
	class gr_edgelist
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;

		typedef TDir directed_type;

	public:

		template<typename ASrc, typename ATar>
		gr_edgelist(gr_size_t nv, gr_size_t ne, ASrc srcs, ATar tars)
		: m_nvertices(nv), m_nedges(ne)
		, m_sources(srcs, ne), m_targets(tars, ne)
		{
		}

	public:

		// basic info

		gr_size_t nvertices() const
		{
			return m_nvertices;
		}

		gr_size_t nedges() const
		{
			return m_nedges;
		}

		vertex_type source_of(const edge_type& e) const
		{
			return m_sources[e.index];
		}

		vertex_type target_of(const edge_type& e) const
		{
			return m_targets[e.index];
		}

		edge_i get_edge_i(const edge_type &e) const
		{
			return edge_i(m_sources[e.index], m_targets[e.index]);
		}


		// iteration

		vertex_iterator v_begin() const
		{
			return make_simple_vertex_iterator(0);
		}

		vertex_iterator v_end() const
		{
			return make_simple_vertex_iterator((gr_index_t)m_nvertices);
		}

		edge_iterator e_begin() const
		{
			return make_simple_edge_iterator(0);
		}

		edge_iterator e_end() const
		{
			return make_simple_edge_iterator((gr_index_t)m_nedges);
		}


	protected:
		gr_size_t m_nvertices;
		gr_size_t m_nedges;

		const_memory_proxy<vertex_type> m_sources;
		const_memory_proxy<vertex_type> m_targets;

	}; // end class gr_edgelist


	template<typename TWeight, typename TDir>
	class gr_wedgelist : public gr_edgelist<TDir>
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;

		typedef TWeight weight_type;
		typedef TDir directed_type;

	public:

		template<typename ASrc, typename ATar, typename AWeight>
		gr_wedgelist(gr_size_t nv, gr_size_t ne, ASrc srcs, ATar tars, AWeight ws)
		: gr_edgelist<TDir>(nv, ne, srcs, tars)
		, m_weights(ws, ne)
		{
		}

		weight_type weight_of(const edge_type& e) const
		{
			return m_weights[e.index];
		}

		wedge_i<weight_type> get_wedge_i(const edge_type& e) const
		{
			return make_edge_i(
					this->m_sources[e.index],
					this->m_targets[e.index],
					this->m_weights[e.index]);
		}


		const weight_type *weights() const
		{
			return m_weights.pbase();
		}


	protected:
		const_memory_proxy<weight_type> m_weights;

	}; // end class gr_wedgelist


}

#endif
