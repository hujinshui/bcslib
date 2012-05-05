/*
 * @file repeat_vectors.h
 *
 * The expression as a facet of repeated rows/columns
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REPEAT_VECTORS_H_
#define BCSLIB_REPEAT_VECTORS_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/bits/repeat_vectors_internal.h>

namespace bcs
{

	// forward declaration

	template<class Arg, int CTCols> class repeat_cols_expr;
	template<class Arg, int CTRows> class repeat_rows_expr;

	/********************************************
	 *
	 *  Generic expression classes
	 *
	 ********************************************/

	template<class Arg, int CTCols>
	struct matrix_traits<repeat_cols_expr<Arg, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<Arg>::value;
		static const int compile_time_num_cols = CTCols;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg, int CTCols>
	class repeat_cols_expr
	: public IMatrixXpr<repeat_cols_expr<Arg, CTCols>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

	public:
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		repeat_cols_expr(const Arg& a, const index_t n)
		: m_arg(a), m_ncols(n)
		{
			check_arg(bcs::is_column(a), "The argument to repeat_cols_expr must be a column.");
		}

		BCS_ENSURE_INLINE const Arg& column() const
		{
			return m_arg;
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return nrows() * ncolumns();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_arg.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

	private:
		const Arg& m_arg;
		const index_t m_ncols;
	};



	template<class Arg, int CTRows>
	struct matrix_traits<repeat_rows_expr<Arg, CTRows> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = ct_cols<Arg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg, int CTRows>
	class repeat_rows_expr
	: public IMatrixXpr<repeat_rows_expr<Arg, CTRows>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

	public:
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		repeat_rows_expr(const Arg& a, const index_t m)
		: m_arg(a), m_nrows(m)
		{
			check_arg(bcs::is_row(a), "The argument to repeat_rows_expr must be a row.");
		}

		BCS_ENSURE_INLINE const Arg& row() const
		{
			return m_arg;
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return nrows() * ncolumns();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_arg.ncolumns();
		}

	private:
		const Arg& m_arg;
		const index_t m_nrows;
	};


	// convenient functions

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	repeat_cols_expr<Arg, DynamicDim> repeat_cols(const IMatrixXpr<Arg, T>& arg, const index_t n)
	{
		return repeat_cols_expr<Arg, DynamicDim>(arg.derived(), n);
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	repeat_rows_expr<Arg, DynamicDim> repeat_rows(const IMatrixXpr<Arg, T>& arg, const index_t m)
	{
		return repeat_rows_expr<Arg, DynamicDim>(arg.derived(), m);
	}


	template<int CTCols>
	struct repcols
	{
		template<class Arg>
		BCS_ENSURE_INLINE
		static repeat_cols_expr<Arg, CTCols> of(
				const IMatrixXpr<Arg, typename matrix_traits<Arg>::value_type>& arg)
		{
			return repeat_cols_expr<Arg, CTCols>(arg.derived(), CTCols);
		}
	};


	template<int CTRows>
	struct reprows
	{
		template<class Arg>
		BCS_ENSURE_INLINE
		static repeat_rows_expr<Arg, CTRows> of(
				const IMatrixXpr<Arg, typename matrix_traits<Arg>::value_type>& arg)
		{
			return repeat_rows_expr<Arg, CTRows>(arg.derived(), CTRows);
		}
	};



	/********************************************
	 *
	 *  vector-wise readers
	 *
	 ********************************************/

	template<class Arg, int CTCols>
	class repcols_colreaders
	: public IVecReaderBank<repcols_colreaders<Arg, CTCols>,
	  	  typename matrix_traits<Arg>::value_type>
	, private noncopyable
	{
	public:
		typedef repeat_cols_expr<Arg, CTCols> expr_type;
		typedef typename expr_type::value_type value_type;

	public:
		BCS_ENSURE_INLINE
		repcols_colreaders(const expr_type& expr)
		: m_in(expr.column())
		{
		}

	public:
		struct reader_type
		: public IVecReader<reader_type, value_type>, private noncopyable
		{
			BCS_ENSURE_INLINE
			reader_type(const repcols_colreaders& host, const index_t)
			: m_in(host.m_in)
			{
			}

			BCS_ENSURE_INLINE value_type get(const index_t i) const
			{
				return m_in.get(i);
			}

		private:
			const typename vec_reader<Arg>::type& m_in;
		};

	private:
		const typename vec_reader<Arg>::type m_in;
	};


	template<class Arg, int CTRows>
	class reprows_colreaders
	: public IVecReaderBank<reprows_colreaders<Arg, CTRows>,
	  	  typename matrix_traits<Arg>::value_type>
	, private noncopyable
	{
	public:
		typedef repeat_rows_expr<Arg, CTRows> expr_type;
		typedef typename expr_type::value_type value_type;

	public:
		BCS_ENSURE_INLINE
		reprows_colreaders(const expr_type& expr)
		: m_in(expr.row())
		{
		}

	public:
		struct reader_type
		: public IVecReader<reader_type, value_type>, private noncopyable
		{
			BCS_ENSURE_INLINE
			reader_type(const reprows_colreaders& host, const index_t j)
			: m_val(host.m_in.get(j))
			{
			}

			BCS_ENSURE_INLINE value_type get(const index_t i) const
			{
				return m_val;
			}

		private:
			value_type m_val;
		};

	private:
		const typename vec_reader<Arg>::type m_in;
	};


	template<class Arg, int CTCols>
	struct colwise_reader_bank<repeat_cols_expr<Arg, CTCols> >
	{
		typedef repcols_colreaders<Arg, CTCols> type;
	};

	template<class Arg, int CTRows>
	struct colwise_reader_bank<repeat_rows_expr<Arg, CTRows> >
	{
		typedef reprows_colreaders<Arg, CTRows> type;
	};


	template<class Arg, int CTCols>
	struct vecacc_cost<repeat_cols_expr<Arg, CTCols>, by_columns_tag>
	{
		static const int value = DenseByColumnAccessCost;
	};

	template<class Arg, int CTCols>
	struct vecacc_cost<repeat_cols_expr<Arg, CTCols>, by_short_columns_tag>
	{
		static const int value = DenseByShortColumnAccessCost;
	};

	template<class Arg, int CTRows>
	struct vecacc_cost<repeat_rows_expr<Arg, CTRows>, by_columns_tag>
	{
		static const int value = DenseByColumnAccessCost;
	};

	template<class Arg, int CTRows>
	struct vecacc_cost<repeat_rows_expr<Arg, CTRows>, by_short_columns_tag>
	{
		static const int value = DenseByShortColumnAccessCost;
	};



	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Arg, int CTCols>
	struct expr_evaluator<repeat_cols_expr<Arg, CTCols> >
	{
		typedef repeat_cols_expr<Arg, CTCols> expr_type;
		typedef typename expr_type::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
		{
			detail::repeat_cols_evaluator<Arg, DMat,
				binary_ct_rows<expr_type, DMat>::value>::evaluate(
						expr.column(), expr.ncolumns(), dst.derived());
		}
	};

	template<class Arg, int CTRows>
	struct expr_evaluator<repeat_rows_expr<Arg, CTRows> >
	{
		typedef repeat_rows_expr<Arg, CTRows> expr_type;
		typedef typename expr_type::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
		{
			detail::repeat_rows_evaluator<Arg, DMat,
				binary_ct_rows<expr_type, DMat>::value>::evaluate(
						expr.row(), expr.nrows(), dst.derived());
		}
	};



}

#endif 
