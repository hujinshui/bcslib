/**
 * @file gr_adjlist.h
 *
 * The class to represent a graph as adjacency list
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_GR_ADJLIST_H
#define BCSLIB_GR_ADJLIST_H

#include <bcslib/graph/graph_base.h>


namespace bcs
{
	template<typename TDir>
	struct gr_adjlist_aux
	{
		static gr_size_t edge_list_size(gr_size_t ne)
		{
			return ne;
		}

		static void do_clone_edgelist(gr_size_t ne, const vertex_t *srcs_in, const vertex_t *tars_in, vertex_t *srcs, vertex_t *tars)
		{
			copy_elements(srcs_in, srcs, ne);
			copy_elements(tars_in, tars, ne);
		}

		template<typename TWeight>
		static void do_clone_edgeweights(gr_size_t ne, const TWeight *ws_in, TWeight *ws)
		{
			copy_elements(ws_in, ws, ne);
		}
	};

	template<>
	struct gr_adjlist_aux<gr_undirected>
	{
		static gr_size_t edge_list_size(gr_size_t ne)
		{
			return 2 * ne;
		}

		static void do_clone_edgelist(gr_size_t ne, const vertex_t *srcs_in, const vertex_t *tars_in, vertex_t *srcs, vertex_t *tars)
		{
			copy_elements(srcs_in, srcs, ne);
			copy_elements(tars_in, srcs + ne, ne);

			copy_elements(tars_in, tars, ne);
			copy_elements(srcs_in, tars + ne, ne);
		}

		template<typename TWeight>
		static void do_clone_edgeweights(gr_size_t ne, const TWeight *ws_in, TWeight *ws)
		{
			copy_elements(ws_in, ws, ne);
			copy_elements(ws_in, ws + ne, ne);
		}
	};





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
		: m_nvertices(nv), m_nedges(ne), m_el(gr_adjlist_aux<TDir>::edge_list_size(ne))
		, m_sources(ref_t(), m_el, srcs), m_targets(ref_t(), m_el, tars)
		, m_out_degrees(ref_t(), nv, degs), m_out_offsets(ref_t(), nv, osets)
		, m_out_neighbors(ref_t(), m_el, nbs), m_out_edges(ref_t(), m_el, adj_es)
		{
		}

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars)
		: m_nvertices(nv), m_nedges(ne), m_el(gr_adjlist_aux<TDir>::edge_list_size(ne))
		, m_sources(ref_t(), m_el, srcs), m_targets(ref_t(), m_el, tars)
		{
			_init_neighbor_structure();
		}

		gr_adjlist(gr_size_t nv, gr_size_t ne,
				clone_t, const vertex_type *srcs, const vertex_type *tars)
		: m_nvertices(nv), m_nedges(ne), m_el(gr_adjlist_aux<TDir>::edge_list_size(ne))
		, m_sources(new block<vertex_type>(m_el)), m_targets(new block<vertex_type>(m_el))
		{
			gr_adjlist_aux<TDir>::do_clone_edgelist(ne, srcs, tars,
					const_cast<vertex_t*>(m_sources.pbase()), const_cast<vertex_t*>(m_targets.pbase()) );

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

		gr_size_t out_degree(const vertex_type& v) const
		{
			return m_out_degrees[v.index];
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

		neighbor_iterator out_neighbor_begin(const vertex_t& v) const
		{
			return m_out_neighbors.pbase() + out_offset_begin(v);
		}

		neighbor_iterator out_neighbor_end(const vertex_t& v) const
		{
			return m_out_neighbors.pbase() + out_offset_end(v);
		}

		adj_edge_iterator out_edge_begin(const vertex_t& v) const
		{
			return m_out_edges.pbase() + out_offset_begin(v);
		}

		adj_edge_iterator out_edge_end(const vertex_t& v) const
		{
			return m_out_edges.pbase() + out_offset_end(v);
		}


	protected:
		gr_index_t out_offset_begin(const vertex_t& v) const
		{
			return m_out_offsets[v.index];
		}

		gr_index_t out_offset_end(const vertex_t& v) const
		{
			return m_out_offsets[v.index] + m_out_degrees[v.index];
		}


	protected:
		gr_size_t m_nvertices;
		gr_size_t m_nedges;
		gr_size_t m_el;     // the length of entire edge list (e.g. for undirected graph, it is 2 * ne)

		const_memory_proxy<vertex_type> m_sources;
		const_memory_proxy<vertex_type> m_targets;

		const_memory_proxy<gr_size_t> m_out_degrees;
		const_memory_proxy<gr_index_t> m_out_offsets;
		const_memory_proxy<vertex_t> m_out_neighbors;
		const_memory_proxy<edge_t> m_out_edges;


	private:
		void _init_neighbor_structure()
		{
			gr_size_t n = m_nvertices;
			gr_size_t m = m_el;

			block<gr_size_t> *p_degs = new block<gr_size_t>(n);
			block<gr_index_t> *p_osets = new block<gr_index_t>(n);
			block<vertex_t> *p_nbs = new block<vertex_t>(m);
			block<edge_t> *p_aes = new block<edge_t>(m);

			const vertex_t *srcs = m_sources.pbase();
			const vertex_t *tars = m_targets.pbase();

			gr_size_t *degs = p_degs->pbase();
			gr_index_t *osets = p_osets->pbase();
			vertex_t *nbs = p_nbs->pbase();
			edge_t *aes = p_aes->pbase();

			// 1st pass: count degrees

			set_zeros_to_elements(degs, n);
			for (gr_size_t e = 0; e < m; ++e)
			{
				++ degs[srcs[e].index];
			}

			// 2nd pass: set offsets

			gr_index_t o = 0;
			for (gr_size_t v = 0; v < n; ++v)
			{
				osets[v] = o;
				o += degs[v];
			}

			// 3rd pass: fill in adjacency info

			for (gr_size_t e = 0; e < m; ++e)
			{
				const vertex_t& sv = srcs[e];
				const vertex_t& tv = tars[e];

				gr_index_t& o = osets[sv.index];

				nbs[o] = tv;
				aes[o] = (gr_index_t)e;

				++o;
			}

			// 4th pass: reset the offsets

			for (gr_size_t v = 0; v < n; ++v)
			{
				osets[v] -= (gr_index_t)degs[v];
			}

			m_out_degrees.reset(p_degs);
			m_out_offsets.reset(p_osets);
			m_out_neighbors.reset(p_nbs);
			m_out_edges.reset(p_aes);
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
		, m_weights(ref_t(), this->m_el, ws)
		{
		}

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				ref_t, const vertex_type *srcs, const vertex_type *tars, const weight_type *ws)
		: gr_adjlist<TDir>(nv, ne, ref_t(), srcs, tars)
		, m_weights(ref_t(), this->m_el, ws)
		{
		}

		gr_wadjlist(gr_size_t nv, gr_size_t ne,
				clone_t, const vertex_type *srcs, const vertex_type *tars, const weight_type *ws)
		: gr_adjlist<TDir>(nv, ne, ref_t(), srcs, tars)
		, m_weights(new block<weight_type>(this->m_el))
		{
			gr_adjlist_aux<TDir>::do_clone_edgeweights(ne, ws,
					const_cast<weight_type*>(m_weights.pbase()));
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

		const weight_type *weights() const
		{
			return m_weights.pbase();
		}

	protected:
		const_memory_proxy<weight_type> m_weights;

	}; // end class gr_wadjlist



}

#endif 
