/*
 * @file repeat_vectors_internal.h
 *
 * The internal implementation of repeat-vectors
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REPEAT_VECTORS_INTERNAL_H_
#define BCSLIB_REPEAT_VECTORS_INTERNAL_H_

#include <bcslib/matrix/vector_proxy.h>
#include <bcslib/matrix/matrix_capture.h>

namespace bcs { namespace detail {

	template<class Arg, int CTCols>
	struct repeat_cols_vecwise
	{
	public:
		typedef typename matrix_traits<Arg>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit repeat_cols_vecwise(const Arg& col)
		: m_cap(col), m_reader(m_cap.get())
		{
		}

		BCS_ENSURE_INLINE value_type load_scalar(index_t i) const
		{
			return m_reader.load_scalar(i);
		}

	private:
		static const bool CanDirectRef = bcs::has_matrix_interface<Arg, IDenseMatrix>::value;
		typedef matrix_capture<Arg, CanDirectRef> capture_t;

		capture_t m_cap;
		vec_reader<typename capture_t::captured_type> m_reader;
	};


	template<class Arg, int CTRows>
	struct repeat_rows_vecwise
	{
	public:
		typedef typename matrix_traits<Arg>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit repeat_rows_vecwise(const Arg& row)
		: m_cap(row), m_icol(0)
		{
		}

		BCS_ENSURE_INLINE value_type load_scalar(index_t i) const
		{
			return m_cap.get()[m_icol];
		}

		BCS_ENSURE_INLINE void operator ++ ()
		{
			++m_icol;
		}

		BCS_ENSURE_INLINE void operator -- ()
		{
			--m_icol;
		}

		BCS_ENSURE_INLINE void operator += (index_t n)
		{
			m_icol += n;
		}

		BCS_ENSURE_INLINE void operator -= (index_t n)
		{
			m_icol -= n;
		}

	private:
		static const bool CanDirectRef =
				bcs::has_matrix_interface<Arg, IDenseMatrix>::value &&
				bcs::matrix_traits<Arg>::is_linear_indexable;

		typedef matrix_capture<Arg, CanDirectRef> capture_t;

		capture_t m_cap;
		index_t m_icol;
	};



	template<class Arg, int CTCols, class DMat>
	struct repeat_cols_evaluator
	{
		typedef typename matrix_traits<Arg>::value_type value_type;

		static const bool CanDirectRef = bcs::has_matrix_interface<Arg, IDenseMatrix>::value;
		typedef matrix_capture<Arg, CanDirectRef> capture_t;

		static void evaluate(const Arg& col, DMat& dst)
		{
			capture_t cap(col);

			vec_reader<typename capture_t::captured_type> rdr(cap.get());
			vecwise_writer<DMat> wrt(dst);

			index_t m = dst.nrows();
			index_t n = dst.ncolumns();

			if (n == 1)
			{
				copy_vec(m, rdr, wrt);
			}
			else
			{
				for (index_t j = 0; j < n; ++j, ++wrt) copy_vec(m, rdr, wrt);
			}

		}
	};


	template<class Arg, int CTRows, class DMat>
	struct repeat_rows_evaluator
	{
		typedef typename matrix_traits<Arg>::value_type value_type;

		static const bool CanDirectRef = matrix_traits<DMat>::is_linear_indexable;

		static void evaluate(const Arg& row, DMat& dst)
		{
			matrix_capture<Arg, CanDirectRef> cap(row);
			typename matrix_capture<Arg, CanDirectRef>::captured_type src = cap.get();

			index_t m = dst.nrows();
			index_t n = dst.ncolumns();

			vecwise_writer<DMat> wrt(dst);

			for (index_t j = 0; j < n; ++j, ++wrt)
			{
				fill_vec(m, wrt, src[j]);
			}
		}
	};



} }


#endif 

