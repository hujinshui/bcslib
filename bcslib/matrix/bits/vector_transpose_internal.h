/**
 * @file vector_transpose_internal.h
 *
 * Internal implementation of vector transpose
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_TRANSPOSE_INTERNAL_H_
#define BCSLIB_VECTOR_TRANSPOSE_INTERNAL_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs { namespace detail {

	struct vec_trans_continuous_tag { };
	struct vec_trans_linear_tag { };
	struct vec_trans_view_tag { };

	template<class Mat>
	struct vec_trans_tag
	{
		typedef typename select_type<matrix_traits<Mat>::is_continuous,
					vec_trans_continuous_tag,
					typename select_type<matrix_traits<Mat>::is_linear_indexable,
						vec_trans_linear_tag,
						vec_trans_view_tag
					>::type
				>::type type;
	};


	template<class SMat, typename STag, bool IsColToRow> struct vec_trans_reader;

	template<class SMat, bool IsColToRow>
	struct vec_trans_reader<SMat, vec_trans_continuous_tag, IsColToRow>
	{
		typedef typename matrix_traits<SMat>::value_type value_type;
		BCS_ENSURE_INLINE static value_type get(const SMat& a, const index_t i)
		{
			return a.ptr_data()[i];
		}
	};

	template<class SMat, bool IsColToRow>
	struct vec_trans_reader<SMat, vec_trans_linear_tag, IsColToRow>
	{
		typedef typename matrix_traits<SMat>::value_type value_type;
		BCS_ENSURE_INLINE static value_type get(const SMat& a, const index_t i)
		{
			return a[i];
		}
	};


	template<class SMat>
	struct vec_trans_reader<SMat, vec_trans_view_tag, true>
	{
		typedef typename matrix_traits<SMat>::value_type value_type;
		BCS_ENSURE_INLINE static value_type get(const SMat& a, const index_t i)
		{
			return a.elem(i, 0);
		}
	};


	template<class SMat>
	struct vec_trans_reader<SMat, vec_trans_view_tag, false>
	{
		typedef typename matrix_traits<SMat>::value_type value_type;

		BCS_ENSURE_INLINE static value_type get(const SMat& a, const index_t i)
		{
			return a(0, i);
		}
	};


	template<class DMat, typename DTag, bool IsColToRow> struct vec_trans_writer;

	template<class DMat, bool IsColToRow>
	struct vec_trans_writer<DMat, vec_trans_continuous_tag, IsColToRow>
	{
		typedef typename matrix_traits<DMat>::value_type value_type;
		BCS_ENSURE_INLINE static void set(DMat& a, const index_t i, const value_type& v)
		{
			a.ptr_data()[i] = v;
		}
	};

	template<class DMat, bool IsColToRow>
	struct vec_trans_writer<DMat, vec_trans_linear_tag, IsColToRow>
	{
		typedef typename matrix_traits<DMat>::value_type value_type;
		BCS_ENSURE_INLINE static void set(DMat& a, const index_t i, const value_type& v)
		{
			a[i] = v;
		}
	};

	template<class DMat>
	struct vec_trans_writer<DMat, vec_trans_view_tag, true>
	{
		typedef typename matrix_traits<DMat>::value_type value_type;
		BCS_ENSURE_INLINE static void set(DMat& a, const index_t i, const value_type& v)
		{
			a.elem(0, i) = v;
		}
	};

	template<class DMat>
	struct vec_trans_writer<DMat, vec_trans_view_tag, false>
	{
		typedef typename matrix_traits<DMat>::value_type value_type;
		BCS_ENSURE_INLINE static void set(DMat& a, const index_t i, const value_type& v)
		{
			a.elem(i, 0) = v;
		}
	};



	template<class SMat, class DMat, typename STag, typename DTag>
	struct vec_trans_evaluator
	{
		BCS_ENSURE_INLINE static void run(const SMat& src, DMat& dst)
		{
			index_t n = src.nelems();

			if (is_column(src))
			{
				for (index_t i = 0; i < n; ++i)
				{
					vec_trans_writer<DMat, DTag, true>::set(dst, i,
							vec_trans_reader<SMat, STag, true>::get(src, i));
				}
			}
			else
			{
				for (index_t i = 0; i < n; ++i)
				{
					vec_trans_writer<DMat, DTag, false>::set(dst, i,
							vec_trans_reader<SMat, STag, false>::get(src, i));
				}
			}
		}
	};

	template<class SMat, class DMat>
	struct vec_trans_evaluator<SMat, DMat, vec_trans_continuous_tag, vec_trans_continuous_tag>
	{
		BCS_ENSURE_INLINE static void run(const SMat& src, DMat& dst)
		{
			index_t n = src.nelems();
			copy_elems(n, src.ptr_data(), dst.ptr_data());
		}
	};



} }


#endif /* VECTOR_TRANSPOSE_INTERNAL_H_ */
