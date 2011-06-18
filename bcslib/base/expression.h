/**
 * @file expression.h
 *
 * The facilities for constructing lazy-evaluation expressions
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_EXPRESSION_H
#define BCSLIB_EXPRESSION_H

#include <bcslib/base/basic_defs.h>
#include <type_traits>
#include <functional>

namespace bcs
{

	namespace _detail
	{

		template<typename T> class operand;

		template<class T>
		class operand<T&&>
		{
		public:
			typedef T value_type;

			operand(T&& x) : m_val(std::move(x)) { }

			operand(operand&& r) : m_val(std::move(r.m_val)) { }

			const value_type& get() const { return m_val; }

		private:
			operand(const operand& );
			operand& operator = (const operand&& );

		private:
			T m_val;
		};


		template<class T>
		class operand<const T&>
		{
		public:
			typedef T value_type;

			operand(const T& x) : m_cref(x) { }

			operand(operand&& r) : m_cref(r.m_cref) { }

			const value_type& get() const { return m_cref; }

		private:
			operand(const operand& );
			operand& operator = (const operand&& );

		private:
			const T& m_cref;
		};
	}


	/******************************************************
	 *
	 *  Expression & Evaluation basis
	 *
	 ******************************************************/

	// unary

	template<typename Func, class T1> class unary_evaluation;

	template<typename Func, class TOperand1>
	class unary_expression : private noncopyable
	{
	public:
		typedef Func func_type;

		typedef TOperand1 op1_type;

		typedef typename op1_type::value_type op1_value_type;

		typedef typename func_type::result_type result_type;

	public:
		unary_expression(func_type f, op1_type&& op1)
		: m_fun(f)
		, m_op1(std::move(op1))
		{
		}

		unary_expression(unary_expression&& r)
		: m_op1(std::move(r.m_op1))
		{
		}

		const op1_value_type& get_op1() const
		{
			return m_op1.get();
		}

		result_type evaluate() const
		{
			return m_fun(get_op1());
		}

	private:
		func_type m_fun;
		op1_type m_op1;
	};


	template<typename Func, class TOperand1>
	inline typename Func::result_type
	evaluate(const unary_expression<Func, TOperand1>& expr)
	{
		return expr.evaluate();
	}


	// binary

	template<typename Func, class T1, class T2> class binary_evaluation;

	template<typename Func, class TOperand1, class TOperand2>
	class binary_expression : private noncopyable
	{
	public:
		typedef Func func_type;

		typedef TOperand1 op1_type;
		typedef TOperand2 op2_type;

		typedef typename op1_type::value_type op1_value_type;
		typedef typename op2_type::value_type op2_value_type;

		typedef typename binary_evaluation<operation_tag,
				op1_value_type, op2_value_type>::result_type result_type;

	public:
		binary_expression(func_type f, op1_type&& op1, op2_type&& op2)
		: m_fun(f)
		, m_op1(std::move(op1))
		, m_op2(std::move(op2))
		{
		}

		binary_expression(func_type f, binary_expression&& r)
		: m_fun(f)
		, m_op1(std::move(r.m_op1))
		, m_op2(std::move(r.m_op2))
		{
		}

		const op1_value_type& get_op1() const
		{
			return m_op1.get();
		}

		const op2_value_type& get_op2() const
		{
			return m_op2.get();
		}

		result_type evaluate() const
		{
			return m_fun(get_op1(), get_op2());
		}

	private:
		func_type m_fun;
		op1_type m_op1;
		op2_type m_op2;
	};


	template<typename Func, class TOperand1, class TOperand2>
	inline typename Func::result_type
	evaluate(const binary_expression<Func, TOperand1, TOperand2>& expr)
	{
		return expr.evaluate();
	}

}


#endif 
