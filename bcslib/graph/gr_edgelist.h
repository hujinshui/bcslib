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
	class gr_edgelist_ref
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;

		typedef TDir directed_type;

	public:

		gr_edgelist_ref(gr_size_t nv, gr_size_t ne, const vertex_type *srcs, const vertex_type *tars)
		: m_nvertices(nv), m_nedges(ne), m_sources(srcs), m_targets(tars)
		{
		}

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


	private:
		gr_size_t m_nvertices;
		gr_size_t m_nedges;

		const vertex_type *m_sources;
		const vertex_type *m_targets;

	}; // end class gr_edgelist


	template<typename TWeight, typename TDir>
	class gr_wedgelist_ref : public gr_edgelist_ref<TDir>
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;

		typedef TWeight weight_type;
		typedef TDir directed_type;

	public:

		gr_wedgelist_ref(gr_size_t nv, gr_size_t ne,
				const vertex_type *srcs, const vertex_type *tars, const weight_type *ws)
		: m_nvertices(nv), m_nedges(ne), m_sources(srcs), m_targets(tars), m_weights(ws)
		{
		}

		weight_type weight_of(const edge_type& e) const
		{
			return m_weights[e.index];
		}

		wedge_i<weight_type> get_wedge_i(const edge_type& e) const
		{
			return make_edge_i(m_sources[e.index], m_targets[e.index], m_weights[e.index]);
		}


	private:
		gr_size_t m_nvertices;
		gr_size_t m_nedges;

		const vertex_type *m_sources;
		const vertex_type *m_targets;
		const weight_type *m_weights;

	}; // end class gr_edgelist


}

#endif
