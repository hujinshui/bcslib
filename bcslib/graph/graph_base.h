/*
 * @file graph_base.h
 *
 * The basic definition for graphs
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_GRAPH_BASE_H
#define BCSLIB_GRAPH_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_mem.h>
#include <bcslib/base/iterators.h>

namespace bcs
{

	/***
	 *
	 *  Basic types
	 *
	 */

	typedef size_t gr_size_t;
	typedef ptrdiff_t gr_index_t;

	struct gr_directed { };
	struct gr_undirected { };
	struct gr_bidirected { };

	struct vertex_t
	{
		gr_index_t index;

		vertex_t()
		{
		}

		vertex_t(gr_index_t i) : index(i)
		{
		}

		bool operator == (const vertex_t& rhs) const
		{
			return index == rhs.index;
		}

		bool operator != (const vertex_t& rhs) const
		{
			return !(operator == (rhs));
		}
	};


	struct edge_t
	{
		gr_index_t index;

		edge_t()
		{
		}

		edge_t(gr_index_t i) : index(i)
		{
		}

		bool operator == (const edge_t& rhs) const
		{
			return index == rhs.index;
		}

		bool operator != (const edge_t& rhs) const
		{
			return !(operator == (rhs));
		}
	};


	struct edge_i
	{
		vertex_t source;
		vertex_t target;

		edge_i()
		{
		}

		edge_i(const vertex_t& s, const vertex_t &t)
		: source(s), target(t)
		{
		}

		bool operator == (const edge_i& rhs) const
		{
			return source == rhs.source && target == rhs.target;
		}

		bool operator != (const edge_i& rhs) const
		{
			return !(operator == (rhs));
		}
	};


	template<typename T>
	struct wedge_i
	{
		vertex_t source;
		vertex_t target;
		T weight;

		wedge_i()
		{
		}

		wedge_i(const vertex_t& s, const vertex_t& t, const T& w)
		: source(s), target(t), weight(w)
		{
		}

		bool operator == (const wedge_i<T>& rhs) const
		{
			return source == rhs.source && target == rhs.target && weight == rhs.weight;
		}

		bool operator != (const wedge_i<T>& rhs) const
		{
			return !(operator == (rhs));
		}
	};


	inline edge_i make_edge_i(const vertex_t& s, const vertex_t& t)
	{
		return edge_i(s, t);
	}

	template<typename T>
	inline wedge_i<T> make_edge_i(const vertex_t& s, const vertex_t& t, const T& w)
	{
		return wedge_i<T>(s, t, w);
	}


	/***
	 *
	 * Basic iterators
	 *
	 */


	class simple_vertex_iterator_impl
	{
	public:
		typedef vertex_t value_type;
		typedef const vertex_t& reference;
		typedef const vertex_t* pointer;

	public:
		simple_vertex_iterator_impl() { }

		simple_vertex_iterator_impl(const vertex_t& v) : m_v(v)
		{
		}

		pointer ptr() const
		{
			return &m_v;
		}

		reference ref() const
		{
			return m_v;
		}

		bool operator == (const simple_vertex_iterator_impl& rhs) const
		{
			return m_v == rhs.m_v;
		}

		void move_next()
		{
			++ m_v.index;
		}

		void move_prev()
		{
			-- m_v.index;
		}

	private:
		vertex_t m_v;
	};

	typedef bidirectional_iterator_wrapper<simple_vertex_iterator_impl> simple_vertex_iterator;

	inline simple_vertex_iterator make_simple_vertex_iterator(const vertex_t& v)
	{
		return simple_vertex_iterator_impl(v);
	}


	class simple_edge_iterator_impl
	{
	public:
		typedef edge_t value_type;
		typedef const edge_t& reference;
		typedef const edge_t* pointer;

	public:
		simple_edge_iterator_impl() { }

		simple_edge_iterator_impl(const edge_t& e) : m_e(e)
		{
		}

		pointer ptr() const
		{
			return &m_e;
		}

		reference ref() const
		{
			return m_e;
		}

		bool operator == (const simple_edge_iterator_impl& rhs) const
		{
			return m_e == rhs.m_e;
		}

		void move_next()
		{
			++ m_e.index;
		}

		void move_prev()
		{
			-- m_e.index;
		}

	private:
		edge_t m_e;
	};


	typedef bidirectional_iterator_wrapper<simple_edge_iterator_impl> simple_edge_iterator;

	inline simple_edge_iterator make_simple_edge_iterator(const edge_t& e)
	{
		return simple_edge_iterator_impl(e);
	}

}

#endif
