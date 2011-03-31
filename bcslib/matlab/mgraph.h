/**
 * @file mgraph.h
 *
 * The MATLAB interface for bcs graph classes
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MGRAPH_H
#define BCSLIB_MGRAPH_H

#include <bcslib/matlab/matlab_base.h>
#include <bcslib/matlab/marray.h>

#include <bcslib/graph/gr_edgelist.h>
#include <bcslib/graph/gr_adjlist.h>

namespace bcs
{

namespace matlab
{

	class const_mgraph
	{
	public:

		static bool is_gr_edgelist(const_marray a)
		{
			return a.is_class("gr_edgelist");
		}

		static bool is_gr_adjlist(const_marray a)
		{
			return a.is_class("gr_adjlist");
		}

	public:
		const_mgraph(const mxArray *mx) : m_a(mx) { }

		const_mgraph(const const_marray& m) : m_a(m) { }

		// fields

		char dtype() const
		{
			return m_a.get_property(0, "dtype").get_scalar<char>();
		}

		gr_size_t nv() const
		{
			return m_a.get_property(0, "nv").get_scalar<double>();
		}

		gr_size_t ne() const
		{
			return (gr_size_t)(m_a.get_property(0, "ne").get_scalar<double>());
		}

		const vertex_t* es() const
		{
			return m_a.get_property(0, "es").data<vertex_t>();
		}

		const vertex_t* et() const
		{
			return m_a.get_property(0, "et").data<vertex_t>();
		}


		bool has_weights() const
		{
			return !(m_a.get_property(0, "ew").is_empty());
		}

		mxClassID weight_type() const
		{
			return m_a.get_property(0, "ew").class_id();
		}

		template<typename TWeight>
		const TWeight* ew() const
		{
			return m_a.get_property(0, "ew").data<TWeight>();
		}

		const gr_size_t* o_ds() const
		{
			return m_a.get_property(0, "o_ds").data<gr_size_t>();
		}

		const gr_index_t* o_os() const
		{
			return m_a.get_property(0, "o_os").data<gr_index_t>();
		}

		const edge_t* o_es() const
		{
			return m_a.get_property(0, "o_es").data<edge_t>();
		}

		const vertex_t* o_ns() const
		{
			return m_a.get_property(0, "o_ns").data<vertex_t>();
		}


	private:
		const_marray m_a;

	}; // end class const_mgraph


	// conversion routines


	template<typename TDir>
	inline gr_edgelist<TDir> to_gr_edgelist(const_mgraph g)
	{
		return gr_edgelist<TDir>(g.nv(), g.ne(),
				ref_t(), g.es(), g.et());
	}


	template<typename TWeight, typename TDir>
	inline gr_wedgelist<TWeight, TDir> to_gr_wedgelist(const_mgraph g)
	{
		return gr_wedgelist<TWeight, TDir>(g.nv(), g.ne(),
				ref_t(), g.es(), g.et(), g.ew<TWeight>());
	}


	template<typename TDir>
	inline gr_adjlist<TDir> to_gr_adjlist(const_mgraph g)
	{
		return gr_adjlist<TDir>(g.nv(), g.ne(),
				ref_t(), g.es(), g.et(), g.o_ds(), g.o_os(), g.o_ns(), g.o_es());
	}


	template<typename TWeight, typename TDir>
	inline gr_wadjlist<TWeight, TDir> to_gr_wadjlist(const_mgraph g)
	{
		return gr_wadjlist<TWeight, TDir>(g.nv(), g.ne(),
				ref_t(), g.es(), g.et(), g.ew<TWeight>(), g.o_ds(), g.o_os(), g.o_ns(), g.o_es());
	}


}

}

#endif 
