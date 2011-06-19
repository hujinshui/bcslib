/**
 * @file expression.h
 *
 * The facilities for constructing scoped lazy-evaluation expressions
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SEXPRESSION_H
#define BCSLIB_SEXPRESSION_H

#include <bcslib/base/basic_defs.h>
#include <type_traits>
#include <functional>

namespace bcs
{

	/******************************************************
	 *
	 *  Expression & Evaluation basis
	 *
	 ******************************************************/

	template<typename T>
	struct is_sexpression
	{
		static const bool value = false;
	};


	// The class to hold leaves of the expression tree

	template<class T>
	class soperand
	{
	public:
		typedef T result_type;

		soperand(T&& x) : m_val(std::move(x)) { }

		soperand(soperand&& r) : m_val(std::move(r.m_val)) { }

		const result_type& get() const { return m_val; }

	private:
		soperand(const soperand& );
		soperand& operator = (const soperand& );

	private:
		T m_val;
	};


	template<class T>
	class soperand_cref
	{
	public:
		typedef T result_type;

		soperand_cref(const T& x) : m_cref(x) { }

		soperand_cref(soperand_cref&& r) : m_cref(r.m_cref) { }

		const result_type& get() const { return m_cref; }

	private:
		soperand_cref(const soperand_cref& );
		soperand_cref& operator = (const soperand_cref& );

	private:
		const T& m_cref;
	};


	template<class T>
	struct is_sexpression<soperand<T> >
	{
		static const bool value = true;
	};

	template<class T>
	struct is_sexpression<soperand_cref<T> >
	{
		static const bool value = true;
	};

	template<class T>
	inline const T& evaluate(const soperand<T>& op)
	{
		return op.get();
	}

	template<class T>
	inline const T& evaluate(const soperand_cref<T>& op)
	{
		return op.get();
	}


	template<class T>
	inline soperand<T> forward_operand(T&& x)
	{
		return std::move(x);
	}

	template<class T>
	inline soperand_cref<T> forward_operand(const T& x)
	{
		return x;
	}


	// unary

	/**
	 * Such way of definition looks ugly.
	 *
	 * TODO rewrite using variadic templates, when provided by all main-stream compilers,
	 */
	template<typename Func, class C1=nil_type, class C2=nil_type, class C3=nil_type>
	class sexpression;

	template<typename Func, class C1, class C2, class C3>
	struct is_sexpression<sexpression<Func, C1, C2, C3> >
	{
		static const bool value = true;
	};


	// 0 children

	template<typename Func>
	class sexpression<Func, nil_type, nil_type, nil_type>
	{
	public:
		static const size_t num_children = 0;
		typedef Func func_type;
		typedef typename std::result_of<func_type()>::type result_type;

	public:
		sexpression(func_type f) : m_fun(f) { }

		sexpression(sexpression&& r) : m_fun(r.m_fun) { }

		result_type evaluate() const
		{
			return m_fun();
		}

	private:
		sexpression(const sexpression& );
		sexpression& operator = (const sexpression& );

	private:
		Func m_fun;
	};

	template<typename Func>
	inline auto evaluate(const sexpression<Func>& expr) -> decltype(expr.evaluate())
	{
		return expr.evaluate();
	}


	// 1 child

	template<typename Func, class C1>
	class sexpression<Func, C1, nil_type, nil_type>
	{
	public:
		typedef Func func_type;

		typedef C1 child1_type;
		typedef typename child1_type::result_type child1_result_type;

		typedef typename std::result_of<func_type(child1_result_type)>::type result_type;

	public:
		sexpression(func_type f, child1_type&& c1)
		: m_fun(f)
		, m_c1(std::move(c1))
		{
		}

		sexpression(sexpression&& r)
		: m_fun(r.m_fun)
		, m_c1(std::move(r.m_c1))
		{
		}

		result_type evaluate() const
		{
			return m_fun(m_c1.evaluate());
		}

	private:
		sexpression(const sexpression& );
		sexpression& operator = (const sexpression& );

	private:
		func_type m_fun;
		child1_type m_c1;
	};

	template<typename Func, class C1>
	inline auto evaluate(const sexpression<Func, C1>& expr) -> decltype(expr.evaluate())
	{
		return expr.evaluate();
	}


	// 2 children

	template<typename Func, class C1, class C2>
	class sexpression<Func, C1, C2, nil_type>
	{
	public:
		typedef Func func_type;

		typedef C1 child1_type;
		typedef C2 child2_type;
		typedef typename child1_type::result_type child1_result_type;
		typedef typename child2_type::result_type child2_result_type;

		typedef typename std::result_of<func_type(child1_result_type, child2_result_type)>::type result_type;

	public:
		sexpression(func_type f, child1_type&& c1, child2_type&& c2)
		: m_fun(f)
		, m_c1(std::move(c1))
		, m_c2(std::move(c2))
		{
		}

		sexpression(sexpression&& r)
		: m_fun(r.m_fun)
		, m_c1(std::move(r.m_c1))
		, m_c2(std::move(r.m_c2))
		{
		}

		result_type evaluate() const
		{
			return m_fun(m_c1.evaluate(), m_c2.evaluate());
		}

	private:
		sexpression(const sexpression& );
		sexpression& operator = (const sexpression& );

	private:
		func_type m_fun;
		child1_type m_c1;
		child2_type m_c2;
	};

	template<typename Func, class C1, class C2>
	inline auto evaluate(const sexpression<Func, C1, C2>& expr) -> decltype(expr.evaluate())
	{
		return expr.evaluate();
	}

}


#endif 
