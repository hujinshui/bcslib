/*
 * @file ewise_matrix_eval.h
 *
 * Element-wise matrix evaluation
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_EWISE_MATRIX_EVAL_H_
#define BCSLIB_EWISE_MATRIX_EVAL_H_

#include <bcslib/matrix/bits/ewise_matrix_eval_internal.h>
#include <bcslib/matrix/vector_accessors.h>

namespace bcs
{
	/********************************************
	 *
	 *  vector readers
	 *
	 ********************************************/

	template<typename Fun, class Arg>
	class unary_ewise_linear_reader
	: public IVecReader<unary_ewise_linear_reader<Fun, Arg>, typename Fun::result_type>
	, private noncopyable
	{
	public:
		typedef unary_ewise_expr<Fun, Arg> expr_type;
		typedef typename Fun::result_type value_type;

		BCS_ENSURE_INLINE
		explicit unary_ewise_linear_reader(const expr_type& expr)
		: fun(expr.fun), arg_reader(expr.arg)
		{
		}

		BCS_ENSURE_INLINE
		value_type get(const index_t i) const
		{
			return fun(arg_reader.get(i));
		}

	private:
		Fun fun;
		typename vec_reader<Arg>::type arg_reader;
	};


	template<typename Fun, class LArg, class RArg>
	class binary_ewise_linear_reader
	: public IVecReader<binary_ewise_linear_reader<Fun, LArg, RArg>, typename Fun::result_type>
	, private noncopyable
	{
	public:
		typedef binary_ewise_expr<Fun, LArg, RArg> expr_type;
		typedef typename Fun::result_type value_type;

		BCS_ENSURE_INLINE
		explicit binary_ewise_linear_reader(const expr_type& expr)
		: fun(expr.fun)
		, left_in(expr.left_arg)
		, right_in(expr.right_arg)
		{
		}

		BCS_ENSURE_INLINE
		value_type get(const index_t i) const
		{
			return fun(left_in.get(i), right_in.get(i));
		}

	private:
		Fun fun;
		typename vec_reader<LArg>::type left_in;
		typename vec_reader<RArg>::type right_in;
	};


	template<typename Fun, class Arg>
	class unary_ewise_colreaders
	: public IVecReaderBank<unary_ewise_colreaders<Fun, Arg>, typename Fun::result_type>
	, private noncopyable
	{
	public:
		typedef unary_ewise_expr<Fun, Arg> expr_type;
		typedef typename Fun::result_type value_type;

	private:
		typedef typename colwise_reader_bank<Arg>::type arg_reader_bank_t;
		typedef typename arg_reader_bank_t::reader_type arg_col_reader_t;

	public:
		BCS_ENSURE_INLINE
		explicit unary_ewise_colreaders(const expr_type& expr)
		: m_fun(expr.fun), m_arg_banks(expr.arg)
		{
		}

	public:
		class reader_type : public IVecReader<reader_type, value_type>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const unary_ewise_colreaders& host, const index_t j)
			: m_fun(host.m_fun), m_in(host.m_arg_banks, j)
			{
			}

			BCS_ENSURE_INLINE
			value_type get(const index_t i) const
			{
				return m_fun(m_in.get(i));
			}

		private:
			const Fun& m_fun;
			arg_col_reader_t m_in;
		};

	private:
		Fun m_fun;
		arg_reader_bank_t m_arg_banks;
	};


	template<typename Fun, class LArg, class RArg>
	class binary_ewise_colreaders
	: public IVecReaderBank<binary_ewise_colreaders<Fun, LArg, RArg>, typename Fun::result_type>
	, private noncopyable
	{
	public:
		typedef binary_ewise_expr<Fun, LArg, RArg> expr_type;
		typedef typename Fun::result_type value_type;

	private:
		typedef typename colwise_reader_bank<LArg>::type left_arg_reader_bank_t;
		typedef typename colwise_reader_bank<RArg>::type right_arg_reader_bank_t;

		typedef typename left_arg_reader_bank_t::reader_type left_arg_col_reader_t;
		typedef typename right_arg_reader_bank_t::reader_type right_arg_col_reader_t;

	public:
		BCS_ENSURE_INLINE
		explicit binary_ewise_colreaders(const expr_type& expr)
		: m_fun(expr.fun)
		, m_left_arg_banks(expr.left_arg)
		, m_right_arg_banks(expr.right_arg)
		{
		}

	public:
		class reader_type : public IVecReader<reader_type, value_type>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const binary_ewise_colreaders& host, const index_t j)
			: m_fun(host.m_fun)
			, m_left_in(host.m_left_arg_banks, j)
			, m_right_in(host.m_right_arg_banks, j)
			{
			}

			BCS_ENSURE_INLINE
			value_type get(const index_t i) const
			{
				return m_fun(m_left_in.get(i), m_right_in.get(i));
			}

		private:
			const Fun& m_fun;
			left_arg_col_reader_t m_left_in;
			right_arg_col_reader_t m_right_in;
		};

	private:
		Fun m_fun;
		left_arg_reader_bank_t m_left_arg_banks;
		right_arg_reader_bank_t m_right_arg_banks;
	};




	/********************************************
	 *
	 *  vector reader dispatch
	 *
	 ********************************************/

	template<class Fun, class Arg, typename Tag>
	struct vecacc_cost<unary_ewise_expr<Fun, Arg>, Tag>
	{
		static const int value = vecacc_cost<Arg, Tag>::value;
	};

	template<class Fun, class LArg, class RArg, typename Tag>
	struct vecacc_cost<binary_ewise_expr<Fun, LArg, RArg>, Tag>
	{
		static const int value =
				vecacc_cost<LArg, Tag>::value +
				vecacc_cost<RArg, Tag>::value;
	};


	template<typename Fun, class Arg>
	struct vec_reader<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_linear_reader<Fun, Arg> type;
	};

	template<typename Fun, class LArg, class RArg>
	struct vec_reader<binary_ewise_expr<Fun, LArg, RArg> >
	{
		typedef binary_ewise_linear_reader<Fun, LArg, RArg> type;
	};

	template<typename Fun, class Arg>
	struct colwise_reader_bank<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_colreaders<Fun, Arg> type;
	};

	template<typename Fun, class LArg, class RArg>
	struct colwise_reader_bank<binary_ewise_expr<Fun, LArg, RArg> >
	{
		typedef binary_ewise_colreaders<Fun, LArg, RArg> type;
	};



	/********************************************
	 *
	 *  evaluation
	 *
	 ********************************************/


	template<typename Fun, class Arg>
	struct expr_evaluator<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_expr<Fun, Arg> expr_type;
		typedef typename expr_type::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			detail::ewise_evaluator<expr_type, DMat,
				is_linear_accessible<DMat>::value>::evaluate(expr, dst.derived());
		}
	};

	template<typename Fun, class LArg, class RArg>
	struct expr_evaluator<binary_ewise_expr<Fun, LArg, RArg> >
	{
		typedef binary_ewise_expr<Fun, LArg, RArg> expr_type;
		typedef typename expr_type::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			detail::ewise_evaluator<expr_type, DMat,
				is_linear_accessible<DMat>::value>::evaluate(expr, dst.derived());
		}
	};

}

#endif 




