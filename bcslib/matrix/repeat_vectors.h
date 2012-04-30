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

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

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

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg, int CTCols>
	class repeat_rows_expr
	: public IMatrixXpr<repeat_rows_expr<Arg, CTCols>, typename matrix_traits<Arg>::value_type>
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



	/********************************************
	 *
	 *  vec-wise proxies
	 *
	 ********************************************/

	template<class Arg, int CTCols>
	class vecwise_reader<repeat_cols_expr<Arg, CTCols> >
	: public IVecReader<vecwise_reader<repeat_cols_expr<Arg, CTCols> >, typename matrix_traits<Arg>::value_type>
	{
	public:
		typedef repeat_cols_expr<Arg, CTCols> expr_type;
		typedef typename matrix_traits<Arg>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vecwise_reader(const expr_type& expr)
		: m_internal(expr.column())
		{
		}

		BCS_ENSURE_INLINE value_type load_scalar(index_t i) const
		{
			return m_internal.load_scalar(i);
		}

		BCS_ENSURE_INLINE void operator ++ () { }

		BCS_ENSURE_INLINE void operator -- () { }

		BCS_ENSURE_INLINE void operator += (index_t n) { }

		BCS_ENSURE_INLINE void operator -= (index_t n) { }

	private:
		detail::repeat_cols_vecwise<Arg, CTCols> m_internal;

	};


	template<class Arg, int CTRows>
	class vecwise_reader<repeat_rows_expr<Arg, CTRows> >
	: public IVecReader<vecwise_reader<repeat_rows_expr<Arg, CTRows> >, typename matrix_traits<Arg>::value_type>
	{
	public:
		typedef repeat_rows_expr<Arg, CTRows> expr_type;
		typedef typename matrix_traits<Arg>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vecwise_reader(const expr_type& expr)
		: m_internal(expr.row())
		{
		}

		BCS_ENSURE_INLINE value_type load_scalar(index_t i) const
		{
			return m_internal.load_scalar(i);
		}

		BCS_ENSURE_INLINE void operator ++ ()
		{
			++ m_internal;
		}

		BCS_ENSURE_INLINE void operator -- ()
		{
			-- m_internal;
		}

		BCS_ENSURE_INLINE void operator += (index_t n)
		{
			m_internal += n;
		}

		BCS_ENSURE_INLINE void operator -= (index_t n)
		{
			m_internal -= n;
		}

	private:
		detail::repeat_rows_vecwise<Arg, CTRows> m_internal;

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
		typedef typename matrix_traits<Arg>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::repeat_cols_evaluator<Arg, binary_ct_cols<expr_type, DMat>::value, DMat>::evaluate(
					expr.column(), dst.derived());
		}
	};

	template<class Arg, int CTRows>
	struct expr_evaluator<repeat_rows_expr<Arg, CTRows> >
	{
		typedef repeat_rows_expr<Arg, CTRows> expr_type;
		typedef typename matrix_traits<Arg>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::repeat_rows_evaluator<Arg, binary_ct_rows<expr_type, DMat>::value, DMat>::evaluate(
					expr.row(), dst.derived());
		}
	};

}

#endif 
