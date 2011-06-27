/**
 * @file generic_array_functions.h
 *
 * A collection of useful array functions based on generci array concept
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_GENERIC_ARRAY_FUNCTIONS_H
#define BCSLIB_GENERIC_ARRAY_FUNCTIONS_H

#include <bcslib/array/array_base.h>
#include <bcslib/base/basic_algorithms.h>
#include <bcslib/base/index_selectors.h>
#include <bcslib/base/arg_check.h>

namespace bcs
{

	/******************************************************
	 *
	 *    Array comparison
	 *
	 ******************************************************/

	template<class LArr, class RArr>
	inline typename std::enable_if<is_array_view<LArr>::value && is_array_view<RArr>::value, bool>::type
	equal_array(const LArr& lhs, const RArr& rhs)
	{
		static_assert(array_view_traits<LArr>::num_dims == array_view_traits<RArr>::num_dims,
				"Two arrays for comparison should have the same number of dimensions.");

		if (get_array_shape(lhs) == get_array_shape(rhs))
		{
			if (is_dense_view(lhs) && is_dense_view(rhs))
			{
				return elements_equal(ptr_base(lhs), ptr_base(rhs), get_num_elems(lhs));
			}
			else
			{
				return std::equal(begin(lhs), end(lhs), begin(rhs));
			}
		}
		else
		{
			return false;
		}
	}


	/******************************************************
	 *
	 *    Export, Import, & Fill
	 *
	 ******************************************************/

	template<class Arr, typename OutputIterator>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	export_to(const Arr& a, OutputIterator dst)
	{
		std::copy_n(begin(a), get_num_elems(a), dst);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	export_to(const Arr& a, typename array_view_traits<Arr>::value_type* dst)
	{
		if (is_dense_view(a))
		{
			copy_elements(ptr_base(a), dst, get_num_elems(a));
		}
		else
		{
			std::copy_n(begin(a), get_num_elems(a), dst);
		}
	}


	template<class Arr, typename InputIterator>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	import_from(Arr& a, InputIterator src)
	{
		std::copy_n(src, get_num_elems(a), begin(a));
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	import_from(Arr& a, const typename array_view_traits<Arr>::value_type* src)
	{
		if (is_dense_view(a))
		{
			copy_elements(src, ptr_base(a), get_num_elems(a));
		}
		else
		{
			std::copy_n(src, get_num_elems(a), begin(a));
		}
	}


	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	fill(Arr& a, const typename array_view_traits<Arr>::value_type& v)
	{
		if (is_dense_view(a))
		{
			std::fill_n(ptr_base(a), get_num_elems(a), v);
		}
		else
		{
			std::fill_n(begin(a), get_num_elems(a), v);
		}
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	set_zeros(Arr& a)
	{
		if (is_dense_view(a))
		{
			set_zeros_to_elements(ptr_base(a), get_num_elems(a));
		}
		else
		{
			throw std::runtime_error("set_zeros can only be applied to dense views.");
		}
	}


	/******************************************************
	 *
	 *    Scoped Proxy
	 *
	 ******************************************************/


	template<class Arr>
	class scoped_aview_read_proxy
	{
	public:
		BCS_STATIC_ASSERT_V(is_array_view<Arr>);
		typedef typename array_view_traits<Arr>::value_type value_type;

		scoped_aview_read_proxy(const Arr& a)
		: is_copy(!is_dense_view(a))
		, m_buf(is_copy ? get_num_elems(a) : size_t(0))
		, m_pbase(is_copy ? m_buf.pbase() : ptr_base(a))
		{
			if (is_copy)
			{
				export_to(a, m_buf.pbase());
			}
		}

		const value_type *pbase()
		{
			return m_pbase;
		}

	private:
		bool is_copy;
		scoped_buffer<value_type> m_buf;
		const value_type *m_pbase;
	};


	template<class Arr>
	class scoped_aview_write_proxy
	{
	public:
		BCS_STATIC_ASSERT_V(is_array_view<Arr>);
		typedef typename array_view_traits<Arr>::value_type value_type;

		scoped_aview_write_proxy(Arr& a, bool use_init_contents)
		: m_rarr(a)
		, is_copy(!is_dense_view(a))
		, m_buf(is_copy ? get_num_elems(a) : size_t(0))
		, m_pbase(is_copy ? m_buf.pbase() : ptr_base(a))
		{
			if (is_copy && use_init_contents)
			{
				export_to(a, m_buf.pbase());
			}
		}

		value_type *pbase()
		{
			return m_pbase;
		}

		void commit()
		{
			if (is_copy)
			{
				import_from(m_rarr, m_pbase);
			}
		}

	private:
		Arr& m_rarr;
		bool is_copy;
		scoped_buffer<value_type> m_buf;
		value_type *m_pbase;
	};


	/******************************************************
	 *
	 *    Sub-array Selection
	 *
	 ******************************************************/

	// select elements from 1D array

	template<class Arr, class IndexSelector>
	inline typename std::enable_if<is_array_view_ndim<Arr, 1>::value,
	typename array_creater<Arr>::result_type>::type
	select_elems(const Arr& a, const IndexSelector& inds)
	{
		typedef typename index_selector_traits<IndexSelector>::input_type input_t;
		typedef typename array_view_traits<Arr>::value_type value_t;

		input_t n = static_cast<input_t>(inds.size());
		auto r = array_creater<Arr>::create(arr_shape((index_t)n));
		value_t *rp = ptr_base(r);

		for (input_t i = 0; i < n; ++i)
		{
			*(rp++) = get(a, inds[i]);
		}

		return r;
	}


	// select elements from 2D array

	template<class Arr, class IndexSelector>
	inline typename std::enable_if<is_array_view_ndim<Arr, 2>::value,
	typename dim_changed_array<Arr, 1>::type>::type
	select_elems(const Arr& a, const IndexSelector& Is, const IndexSelector& Js)
	{
		check_arg(Is.size() == Js.size(), "Inconsistent sizes of input index collections.");

		typedef typename index_selector_traits<IndexSelector>::input_type input_t;
		typedef typename array_view_traits<Arr>::value_type value_t;

		input_t n = static_cast<input_t>(Is.size());
		auto r = dim_changed_array<Arr, 1>::create(arr_shape((index_t)n));
		value_t *rp = ptr_base(r);

		for (input_t i = 0; i < n; ++i)
		{
			*(rp++) = get(a, Is[i], Js[i]);
		}

		return r;
	}


	template<class Arr, class IndexSelector>
	inline typename std::enable_if<is_array_view_ndim<Arr, 2>::value,
	typename array_creater<Arr>::result_type>::type
	select_rows(const Arr& a, const IndexSelector& irows)
	{
		typedef typename index_selector_traits<IndexSelector>::input_type input_t;
		typedef typename array_view_traits<Arr>::value_type value_t;
		typedef typename array_view_traits<Arr>::layout_order layout_order;

		auto shape_a = get_array_shape(a);
		index_t d1 = shape_a[1];

		input_t n = static_cast<input_t>(irows.size());
		auto r = array_creater<Arr>::create(arr_shape((index_t)n, d1));
		value_t *rp = ptr_base(r);

		if (std::is_same<layout_order, row_major_t>::value)
		{
			for (input_t i = 0; i < n; ++i)
			{
				index_t ir = irows[i];
				for (index_t j = 0; j < d1; ++j)
				{
					*(rp++) = get(a, ir, j);
				}
			}
		}
		else
		{
			for (index_t j = 0; j < d1; ++j)
			{
				for (input_t i = 0; i < n; ++i)
				{
					*(rp++) = get(a, irows[i], j);
				}
			}
		}

		return r;
	}


	template<class Arr, class IndexSelector>
	inline typename std::enable_if<is_array_view_ndim<Arr, 2>::value,
	typename array_creater<Arr>::result_type>::type
	select_columns(const Arr& a, const IndexSelector& icols)
	{
		typedef typename index_selector_traits<IndexSelector>::input_type input_t;
		typedef typename array_view_traits<Arr>::value_type value_t;
		typedef typename array_view_traits<Arr>::layout_order layout_order;

		auto shape_a = get_array_shape(a);
		index_t d0 = shape_a[0];

		input_t n = static_cast<input_t>(icols.size());
		auto r = array_creater<Arr>::create(arr_shape(d0, (index_t)n));
		value_t *rp = ptr_base(r);

		if (std::is_same<layout_order, row_major_t>::value)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				for (input_t j = 0; j < n; ++j)
				{
					*(rp++) = get(a, i, icols[j]);
				}
			}
		}
		else
		{
			for (input_t j = 0; j < n; ++j)
			{
				index_t jc = icols[j];
				for (index_t i = 0; i < d0; ++i)
				{
					*(rp++) = get(a, i, jc);
				}
			}
		}

		return r;
	}


	template<class Arr, class IndexSelector0, class IndexSelector1>
	inline typename std::enable_if<is_array_view_ndim<Arr, 2>::value,
	typename array_creater<Arr>::result_type>::type
	select_rows_and_cols(const Arr& a, const IndexSelector0& irows, const IndexSelector1& icols)
	{
		typedef typename index_selector_traits<IndexSelector0>::input_type input_t0;
		typedef typename index_selector_traits<IndexSelector1>::input_type input_t1;

		typedef typename array_view_traits<Arr>::value_type value_t;
		typedef typename array_view_traits<Arr>::layout_order layout_order;

		input_t0 m = static_cast<input_t0>(irows.size());
		input_t1 n = static_cast<input_t1>(icols.size());
		auto r = array_creater<Arr>::create(arr_shape((index_t)m, (index_t)n));
		value_t *rp = ptr_base(r);

		if (std::is_same<layout_order, row_major_t>::value)
		{
			for (input_t0 i = 0; i < m; ++i)
			{
				index_t ir = irows[i];
				for (input_t1 j = 0; j < n; ++j)
				{
					*(rp++) = get(a, ir, icols[j]);
				}
			}
		}
		else
		{
			for (input_t1 j = 0; j < n; ++j)
			{
				index_t jc = icols[j];
				for (input_t0 i = 0; i < m; ++i)
				{
					*(rp++) = get(a, irows[i], jc);
				}
			}
		}

		return r;
	}
}

#endif 
