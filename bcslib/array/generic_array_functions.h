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

namespace bcs
{

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


	template<class Arr>
	class scoped_aview_read_proxy
	{
	public:
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

}

#endif 
