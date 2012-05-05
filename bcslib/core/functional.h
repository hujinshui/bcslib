/**
 * @file functional.h
 *
 * The support of functional programming
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_FUNCTIONAL_H_
#define BCSLIB_FUNCTIONAL_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/core/syntax.h>

#define BCS_MAXIMUM_EWISE_ARGUMENTS 4

#define DECLARE_EWISE_FUNCTOR(Name, N) \
	template<typename T> struct is_ewise_functor<Name<T>, N> { static const bool value = true; }; \
	template<typename T> struct num_arguments<Name<T> > { static const bool value = N; };

#define DECLARE_REDUCTOR(Name, N) \
	template<typename T> struct is_reduction_functor<Name<T>, N> { static const bool value = true; }; \
	template<typename T> struct num_arguments<Name<T> > { static const bool value = N; };


namespace bcs
{
	template<typename F> struct num_arguments;


	template<typename F, int N>
	struct is_ewise_functor
	{
		static const bool value = false;
	};

	template<typename F, int N>
	struct is_reductor
	{
		static const bool value = false;
	};


	/********************************************
	 *
	 *  Unary composition
	 *
	 ********************************************/

	template<typename TopF, typename F1, int N1>
	struct unary_compose_fun;


	template<typename TopF, typename F1, int N1>
	struct is_ewise_functor<unary_compose_fun<TopF, F1, N1>, N1>
	{
		static const bool value = true;
	};

	template<typename TopF, typename F1, int N1>
	struct num_arguments<unary_compose_fun<TopF, F1, N1> >
	{
		static const bool value = N1;
	};


	template<typename TopF, typename F1>
	struct unary_compose_fun<TopF, F1, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 1>::value, "TopF should be a unary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		TopF top_fun;
		F1 fun1;

		BCS_ENSURE_INLINE
		unary_compose_fun(const TopF& tf, const F1& f1)
		: top_fun(tf), fun1(f1) { }

		BCS_ENSURE_INLINE
		result_type operator() () const
		{
			return top_fun(fun1());
		}
	};


	template<typename TopF, typename F1>
	struct unary_compose_fun<TopF, F1, 1>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 1>::value, "TopF should be a unary ewise functor");
		static_assert(is_ewise_functor<F1, 1>::value, "F1 should be an ewise functor with 1 arguments");
#endif

		typedef typename TopF::result_type result_type;
		typedef typename F1::arg1_type arg1_type;

		TopF top_fun;
		F1 fun1;

		BCS_ENSURE_INLINE
		unary_compose_fun(const TopF& tf, const F1& f1)
		: top_fun(tf), fun1(f1) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a) const
		{
			return top_fun(fun1(a));
		}
	};


	template<typename TopF, typename F1>
	struct unary_compose_fun<TopF, F1, 2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 1>::value, "TopF should be a unary ewise functor");
		static_assert(is_ewise_functor<F1, 2>::value, "F1 should be an ewise functor with 2 arguments");
#endif

		typedef typename TopF::result_type result_type;
		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;

		TopF top_fun;
		F1 fun1;

		BCS_ENSURE_INLINE
		unary_compose_fun(const TopF& tf, const F1& f1)
		: top_fun(tf), fun1(f1) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1, const arg2_type& a2) const
		{
			return top_fun(fun1(a1, a2));
		}
	};


	template<typename TopF, typename F1>
	struct unary_compose_fun<TopF, F1, 3>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 1>::value, "TopF should be a unary ewise functor");
		static_assert(is_ewise_functor<F1, 3>::value, "F1 should be an ewise functor with 3 arguments");
#endif

		typedef typename TopF::result_type result_type;
		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F1::arg3_type arg3_type;

		TopF top_fun;
		F1 fun1;

		BCS_ENSURE_INLINE
		unary_compose_fun(const TopF& tf, const F1& f1)
		: top_fun(tf), fun1(f1) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3) const
		{
			return top_fun(fun1(a1, a2, a3));
		}
	};


	template<typename TopF, typename F1>
	struct unary_compose_fun<TopF, F1, 4>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 1>::value, "TopF should be a unary ewise functor");
		static_assert(is_ewise_functor<F1, 4>::value, "F1 should be an ewise functor with 4 arguments");
#endif

		typedef typename TopF::result_type result_type;
		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F1::arg3_type arg3_type;
		typedef typename F1::arg4_type arg4_type;

		TopF top_fun;
		F1 fun1;

		BCS_ENSURE_INLINE
		unary_compose_fun(const TopF& tf, const F1& f1)
		: top_fun(tf), fun1(f1) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(a1, a2, a3, a4));
		}
	};


	/********************************************
	 *
	 *  Binary composition
	 *
	 ********************************************/

	template<typename TopF, typename F1, int N1, typename F2, int N2>
	struct binary_compose_fun;

#define BCS_EWISE_BINARY_COMPOSE_TRAITS( N1, N2, Nr ) \
	template<typename TopF, typename F1, typename F2> \
	struct is_ewise_functor<binary_compose_fun<TopF, F1, N1, F2, N2>, Nr> { static const bool value = true; };

	BCS_EWISE_BINARY_COMPOSE_TRAITS(0, 0, 0)

	BCS_EWISE_BINARY_COMPOSE_TRAITS(1, 0, 1)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(0, 1, 1)

	BCS_EWISE_BINARY_COMPOSE_TRAITS(2, 0, 2)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(1, 1, 2)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(0, 2, 2)

	BCS_EWISE_BINARY_COMPOSE_TRAITS(3, 0, 3)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(2, 1, 3)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(1, 2, 3)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(0, 3, 3)

	BCS_EWISE_BINARY_COMPOSE_TRAITS(4, 0, 4)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(3, 1, 4)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(2, 2, 4)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(1, 3, 4)
	BCS_EWISE_BINARY_COMPOSE_TRAITS(0, 4, 4)

#undef BCS_EWISE_BINARY_COMPOSE_TRAITS

	template<typename TopF, typename F1, int N1, typename F2, int N2>
	struct num_arguments<binary_compose_fun<TopF, F1, N1, F2, N2> >
	{
		static const bool value = N1 + N2;
	};


	// N1 + N2 = 0

	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 0, F2, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
		static_assert(is_ewise_functor<F2, 0>::value, "F2 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() () const
		{
			return top_fun(fun1(), fun2());
		}
	};


	// N1 + N2 = 1

	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 1, F2, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 1>::value, "F1 should be an ewise functor with 1 arguments");
		static_assert(is_ewise_functor<F2, 0>::value, "F2 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1) const
		{
			return top_fun(fun1(a1), fun2());
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 0, F2, 1>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
		static_assert(is_ewise_functor<F2, 1>::value, "F2 should be an ewise functor with 1 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F2::arg1_type arg1_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1) const
		{
			return top_fun(fun1(), fun2(a1));
		}
	};


	// N1 + N2 = 2

	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 2, F2, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 2>::value, "F1 should be an ewise functor with 2 arguments");
		static_assert(is_ewise_functor<F2, 0>::value, "F2 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1, const arg2_type& a2) const
		{
			return top_fun(fun1(a1, a2), fun2());
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 1, F2, 1>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 1>::value, "F1 should be an ewise functor with 1 arguments");
		static_assert(is_ewise_functor<F2, 1>::value, "F2 should be an ewise functor with 1 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F2::arg1_type arg2_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1, const arg2_type& a2) const
		{
			return top_fun(fun1(a1), fun2(a2));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 0, F2, 2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
		static_assert(is_ewise_functor<F2, 2>::value, "F2 should be an ewise functor with 2 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F2::arg1_type arg1_type;
		typedef typename F2::arg2_type arg2_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (const arg1_type& a1, const arg2_type& a2) const
		{
			return top_fun(fun1(), fun2(a1, a2));
		}
	};


	// N1 + N2 = 3

	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 3, F2, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 3>::value, "F1 should be an ewise functor with 3 arguments");
		static_assert(is_ewise_functor<F2, 0>::value, "F2 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F1::arg3_type arg3_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3) const
		{
			return top_fun(fun1(a1, a2, a3), fun2());
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 2, F2, 1>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 2>::value, "F1 should be an ewise functor with 2 arguments");
		static_assert(is_ewise_functor<F2, 1>::value, "F2 should be an ewise functor with 1 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F2::arg1_type arg3_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3) const
		{
			return top_fun(fun1(a1, a2), fun2(a3));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 1, F2, 2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 1>::value, "F1 should be an ewise functor with 1 arguments");
		static_assert(is_ewise_functor<F2, 2>::value, "F2 should be an ewise functor with 2 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F2::arg1_type arg2_type;
		typedef typename F2::arg2_type arg3_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3) const
		{
			return top_fun(fun1(a1), fun2(a2, a3));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 0, F2, 3>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
		static_assert(is_ewise_functor<F2, 3>::value, "F2 should be an ewise functor with 3 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F2::arg1_type arg1_type;
		typedef typename F2::arg2_type arg2_type;
		typedef typename F2::arg3_type arg3_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3) const
		{
			return top_fun(fun1(), fun2(a1, a2, a3));
		}
	};


	// N1 + N2 = 4

	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 4, F2, 0>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 4>::value, "F1 should be an ewise functor with 4 arguments");
		static_assert(is_ewise_functor<F2, 0>::value, "F2 should be an ewise functor with 0 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F1::arg3_type arg3_type;
		typedef typename F1::arg4_type arg4_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(a1, a2, a3, a4), fun2());
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 3, F2, 1>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 3>::value, "F1 should be an ewise functor with 3 arguments");
		static_assert(is_ewise_functor<F2, 1>::value, "F2 should be an ewise functor with 1 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F1::arg3_type arg3_type;
		typedef typename F2::arg1_type arg4_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(a1, a2, a3), fun2(a4));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 2, F2, 2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 2>::value, "F1 should be an ewise functor with 2 arguments");
		static_assert(is_ewise_functor<F2, 2>::value, "F2 should be an ewise functor with 2 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F1::arg2_type arg2_type;
		typedef typename F2::arg1_type arg3_type;
		typedef typename F2::arg2_type arg4_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(a1, a2), fun2(a3, a4));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 1, F2, 3>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 1>::value, "F1 should be an ewise functor with 1 arguments");
		static_assert(is_ewise_functor<F2, 3>::value, "F2 should be an ewise functor with 3 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F1::arg1_type arg1_type;
		typedef typename F2::arg1_type arg2_type;
		typedef typename F2::arg2_type arg3_type;
		typedef typename F2::arg3_type arg4_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(a1), fun2(a2, a3, a4));
		}
	};


	template<typename TopF, typename F1, typename F2>
	struct binary_compose_fun<TopF, F1, 0, F2, 4>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_ewise_functor<TopF, 2>::value, "TopF should be a binary ewise functor");
		static_assert(is_ewise_functor<F1, 0>::value, "F1 should be an ewise functor with 0 arguments");
		static_assert(is_ewise_functor<F2, 4>::value, "F2 should be an ewise functor with 4 arguments");
#endif
		typedef typename TopF::result_type result_type;

		typedef typename F2::arg1_type arg1_type;
		typedef typename F2::arg2_type arg2_type;
		typedef typename F2::arg3_type arg3_type;
		typedef typename F2::arg4_type arg4_type;

		TopF top_fun;
		F1 fun1;
		F2 fun2;

		BCS_ENSURE_INLINE
		binary_compose_fun(const TopF& tf, const F1& f1, const F2& f2)
		: top_fun(tf), fun1(f1), fun2(f2) { }

		BCS_ENSURE_INLINE
		result_type operator() (
				const arg1_type& a1,
				const arg2_type& a2,
				const arg3_type& a3,
				const arg4_type& a4) const
		{
			return top_fun(fun1(), fun2(a1, a2, a3, a4));
		}
	};

}

#endif /* FUNCTIONAL_H_ */
