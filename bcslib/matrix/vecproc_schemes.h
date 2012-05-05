/**
 * @file vecproc_schemes.h
 *
 * The scheme to process vectors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECPROC_SCHEMES_H_
#define BCSLIB_VECPROC_SCHEMES_H_

#include <bcslib/core/basic_defs.h>

namespace bcs
{

	// per-scalar operations
	struct vecscheme_by_scalars
	{
		const index_t length;

		BCS_ENSURE_INLINE
		vecscheme_by_scalars(const index_t len) : length(len) { }
	};

	template<int N> struct vecscheme_by_fixed_num_scalars
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(N >= 1, "N must be a positive integer.");
#endif
	};


	/********************************************
	 *
	 * 	vecscheme resolution
	 *
	 ********************************************/

	template<class LArg, class RArg>
	struct binary_nil_expr;

	template<class LArg, class RArg>
	struct matrix_traits<binary_nil_expr<LArg, RArg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = binary_ct_rows<LArg, RArg>::value;
		static const int compile_time_num_cols = binary_ct_cols<LArg, RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	struct binary_nil_expr
	: public IMatrixXpr<binary_nil_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

		const LArg& left_arg;
		const RArg& right_arg;

		binary_nil_expr(const LArg& a1, const RArg& a2)
		: left_arg(a1), right_arg(a2)
		{ }

		BCS_ENSURE_INLINE index_type nelems() const { return left_arg.nelems(); }
		BCS_ENSURE_INLINE size_type size() const { return left_arg.size(); }
		BCS_ENSURE_INLINE index_type nrows() const { return left_arg.nrows(); }
		BCS_ENSURE_INLINE index_type ncolumns() const { return left_arg.ncolumns(); }
	};



	template<class Expr>
	class single_vecscheme
	{
	private:
		struct dyn_vecscheme
		{
			typedef vecscheme_by_scalars type;

			BCS_ENSURE_INLINE
			static type get(const Expr& expr) { return type(expr.nelems()); }
		};

		template<int N>
		struct sta_vecscheme
		{
			typedef vecscheme_by_fixed_num_scalars<N> type;
			BCS_ENSURE_INLINE
			static type get(const Expr& expr) { return type(); }
		};

		typedef typename select_type<has_dynamic_nrows<Expr>::value,
					dyn_vecscheme,
					sta_vecscheme<ct_size<Expr>::value>
				>::type internal_t;
	public:

		typedef typename internal_t::type type;

		BCS_ENSURE_INLINE
		static type get(const Expr& expr) { return internal_t::get(expr); }
	};

	template<class Expr>
	class colwise_vecscheme
	{
	private:
		struct dyn_vecscheme
		{
			typedef vecscheme_by_scalars type;

			BCS_ENSURE_INLINE
			static type get(const Expr& expr) { return type(expr.ncolumns()); }
		};

		template<int N>
		struct sta_vecscheme
		{
			typedef vecscheme_by_fixed_num_scalars<N> type;
			BCS_ENSURE_INLINE
			static type get(const Expr& expr) { return type(); }
		};

		typedef typename select_type<has_dynamic_nrows<Expr>::value,
					dyn_vecscheme,
					sta_vecscheme<ct_rows<Expr>::value>
				>::type internal_t;
	public:

		typedef typename internal_t::type type;

		BCS_ENSURE_INLINE
		static type get(const Expr& expr) { return internal_t::get(expr); }
	};


}

#endif /* VECPROC_SCHEMES_H_ */
