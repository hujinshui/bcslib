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



	// vec-scheme resolve

	template<class Expr>
	class vecscheme_as_single_vec
	{
	private:
		struct dyn_vecscheme
		{
			typedef vecscheme_by_scalars type;

			BCS_ENSURE_INLINE
			static type get(const Expr& expr)
			{
				return type(expr.nelems());
			}
		};

		template<int N>
		struct sta_vecscheme
		{
			typedef vecscheme_by_fixed_num_scalars<N> type;
			BCS_ENSURE_INLINE
			static type get(const Expr& expr)
			{
				return type();
			}
		};

		typedef typename select_type<has_dynamic_nrows,
					dyn_vecscheme,
					sta_vecscheme<ct_size<Expr>::value>
				>::type internal_t;
	public:

		typedef typename internal_t::type type;

		BCS_ENSURE_INLINE
		static type get(const Expr& expr)
		{
			return internal_t::get(expr);
		}
	};

	template<class Expr>
	class vecscheme_by_columns
	{
	private:
		struct dyn_vecscheme
		{
			typedef vecscheme_by_scalars type;

			BCS_ENSURE_INLINE
			static type get(const Expr& expr)
			{
				return type(expr.ncolumns());
			}
		};

		template<int N>
		struct sta_vecscheme
		{
			typedef vecscheme_by_fixed_num_scalars<N> type;
			BCS_ENSURE_INLINE
			static type get(const Expr& expr)
			{
				return type();
			}
		};

		typedef typename select_type<has_dynamic_nrows,
					dyn_vecscheme,
					sta_vecscheme<ct_rows<Expr>::value>
				>::type internal_t;
	public:

		typedef typename internal_t::type type;

		BCS_ENSURE_INLINE
		static type get(const Expr& expr)
		{
			return internal_t::get(expr);
		}
	};


}

#endif /* VECPROC_SCHEMES_H_ */
