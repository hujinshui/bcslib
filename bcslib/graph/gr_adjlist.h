/**
 * @file gr_adjlist.h
 *
 * The class to represent a graph as adjacency list
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_GR_ADJLIST_H
#define BCSLIB_GR_ADJLIST_H

namespace bcs
{

	template<typename TDir>
	class gr_adjlist
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;
		typedef const vertex_type* neighbor_iterator;
		typedef const edge_type* adj_edge_iterator;

		typedef TDir directed_type;

	public:

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars,
				const gr_size_t *degs, const gr_index_t *osets, const vertex_t *nbs, const edge_t *adj_es)
		: m_nvertices(nv), m_nedges(ne)
		, m_sources(ref_t(), ne, srcs), m_targets(ref_t(), ne, tars)
		, m_degrees(ref_t(), nv, degs), m_offsets(ref_t(), nv, osets)
		, m_neighbors(ref_t(), ne, nbs), m_adj_edges(ref_t(), ne, adj_es)
		{
		}

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars)
		: m_nvertices(nv), m_nedges(ne)
		, m_sources(ref_t(), ne, srcs), m_targets(ref_t(), ne, tars)
		{
			_init_neighbor_structure();
		}

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				clone_t, const vertex_type *srcs, const vertex_type *tars)
		: m_nvertices(nv), m_nedges(ne)
		, m_sources(clone_t(), ne, srcs), m_targets(clone_t(), ne, tars)
		{
			_init_neighbor_structure();
		}

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				block<vertex_type>* srcs, block<vertex_type>* tars)
		: m_nvertices(nv), m_nedges(ne)
		, m_sources(srcs), m_targets(tars)
		{
			_init_neighbor_structure();
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

		neighbor_iterator neighbor_begin(const vertex_t& v) const
		{
			return m_neighbors.pbase() + adj_offset_begin(v);
		}

		neighbor_iterator neighbor_end(const vertex_t& v) const
		{
			return m_neighbors.pbase() + adj_offset_end(v);
		}

		adj_edge_iterator adjedge_begin(const vertex_t& v) const
		{
			return m_adj_edges.pbase() + adj_offset_begin(v);
		}

		adj_edge_iterator adjedge_end(const vertex_t& v) const
		{
			return m_adj_edges.pbase() + adj_offset_end(v);
		}


	protected:
		gr_index_t adj_offset_begin(const vertex_t& v) const
		{
			return m_offsets[v.index];
		}

		gr_index_t adj_offset_end(const vertex_t& v) const
		{
			return m_offsets[v.index] + m_degrees[v.index];
		}


	protected:
		gr_size_t m_nvertices;
		gr_size_t m_nedges;

		const_memory_proxy<vertex_type> m_sources;
		const_memory_proxy<vertex_type> m_targets;

		const_memory_proxy<gr_size_t> m_degrees;
		const_memory_proxy<gr_index_t> m_offsets;
		const_memory_proxy<vertex_t> m_neighbors;
		const_memory_proxy<edge_t> m_adj_edges;


	private:
		void _init_neighbor_structure()
		{

		}

	}; // end class gr_adjlist


	template<typename TWeight, typename TDir>
	class gr_wadjlist : public gr_adjlist<TDir>
	{
	public:
		typedef vertex_t vertex_type;
		typedef edge_t edge_type;

		typedef simple_vertex_iterator vertex_iterator;
		typedef simple_edge_iterator edge_iterator;
		typedef const vertex_type* neighbor_iterator;
		typedef const edge_type* adj_edge_iterator;

		typedef TWeight weight_type;
		typedef TDir directed_type;

	public:

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars, const weight_type *ws,
				const gr_size_t *degs, const gr_index_t *osets, const vertex_t *nbs, const edge_t *adj_es)
		: gr_adjlist<TDir>(nv, ne, ref_t(), srcs, tars, degs, osets, nbs, adj_es)
		, m_weights(ref_t(), ne, ws)
		{
		}

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars, const weight_type *ws)
		: gr_adjlist<TDir>(nv, ne, ref_t(), srcs, tars)
		, m_weights(ref_t(), ne, ws)
		{
		}

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				clone_t, const vertex_type *srcs, const vertex_type *tars, const weight_type *ws)
		: gr_adjlist<TDir>(nv, ne, ref_t(), srcs, tars)
		, m_weights(clone_t(), ne, ws)
		{
		}

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				block<vertex_type>* srcs, block<vertex_type>* tars, block<weight_type>* ws)
		: gr_adjlist<TDir>(nv, ne, srcs, tars)
		, m_weights(ws)
		{
		}

	public:

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


	protected:
		const_memory_proxy<weight_type> m_weights;

	}; // end class gr_wadjlist



}

#endif 
