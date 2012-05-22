/**
 * @file matrix_prod.h
 *
 * Matrix product expression and evaluation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PROD_H_
#define BCSLIB_MATRIX_PROD_H_

#include <bcslib/linalg/bits/matrix_prod_internal.h>

namespace bcs
{

	/********************************************
	 *
	 *  Expression classes
	 *
	 ********************************************/

	template<class LArg, class RArg, typename LTag, typename RTag> class mm_expr;

	template<class LArg, class RArg, typename LTag, typename RTag>
	struct matrix_traits<mm_expr<LArg, RArg, LTag, RTag> >
	{
		static const int num_dimensions = 2;

		static const int compile_time_num_rows =
				bcs::is_same<LTag, mm_tmat_tag>::value ? ct_rows<LArg>::value : ct_cols<LArg>::value;

		static const int compile_time_num_cols =
				bcs::is_same<RTag, mm_tmat_tag>::value ? ct_cols<RArg>::value : ct_rows<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};

	template<class LArg, class RArg, typename LTag, typename RTag>
	class mm_expr : public matrix_prod_expr_base<LArg, RArg>
	, public IMatrixXpr<mm_expr<LArg, RArg, LTag, RTag>, typename matrix_traits<LArg>::value_type>
	{
		typedef matrix_prod_expr_base<LArg, RArg> base_t;

	public:
		typedef detail::mm_expr_intern<LArg, RArg, LTag, RTag> intern_t;
		BCS_MAT_TRAITS_CDEFS(typename base_t::value_type)

	public:
		mm_expr(const LArg& larg, const RArg& rarg, const value_type alpha = value_type(1))
		: base_t(larg, rarg, alpha)
		{
			intern_t::check_args(larg, rarg);
		}

		mm_expr(const mm_expr& e0, const value_type alpha)  // replace the alpha value
		: base_t(e0.left_arg(), e0.right_arg(), alpha)
		{
		}

		BCS_ENSURE_INLINE
		index_type nelems() const
		{
			return intern_t::get_nelems(this->left_arg(), this->right_arg());
		}

		BCS_ENSURE_INLINE
		size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE
		index_type nrows() const
		{
			return intern_t::get_nrows(this->left_arg());
		}

		BCS_ENSURE_INLINE
		index_type ncolumns() const
		{
			return intern_t::get_ncols(this->right_arg());
		}
	};


	/********************************************
	 *
	 *  Evaluators
	 *
	 ********************************************/

	template<class LArg, class RArg, typename LTag, typename RTag>
	struct expr_evaluator<mm_expr<LArg, RArg, LTag, RTag> >
	{
		typedef mm_expr<LArg, RArg, LTag, RTag> expr_type;
		typedef typename matrix_traits<expr_type>::value_type T;

		typedef typename expr_type::intern_t intern_t;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value > left(expr.left_arg());
			matrix_capture<RArg, is_dense_mat<RArg>::value > right(expr.right_arg());

			intern_t::eval(expr.alpha(), left.get(), right.get(), T(0), dst.derived());
		}
	};


	/********************************************
	 *
	 *  Expression construction
	 *
	 ********************************************/

	template<class LArg, class RArg>
	struct mm_expr_t
	{
		typedef mm_expr<
				typename detail::mm_left_arg<LArg>::type,
				typename detail::mm_right_arg<RArg>::type,
				typename detail::mm_left_arg<LArg>::tag,
				typename detail::mm_right_arg<RArg>::tag> type;
	};


	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline typename mm_expr_t<LArg, RArg>::type
	mm(const IMatrixXpr<LArg, T>& a, const IMatrixXpr<RArg, T>& b)
	{
		typedef typename mm_expr_t<LArg, RArg>::type result_expr_t;

		return result_expr_t(
				detail::mm_left_arg<LArg>::get(a.derived()),
				detail::mm_right_arg<RArg>::get(b.derived()));
	}

	template<class LArg, class RArg, typename LTag, typename RTag>
	BCS_ENSURE_INLINE
	inline mm_expr<LArg, RArg, LTag, RTag> operator * (
			const mm_expr<LArg, RArg, LTag, RTag>& expr, const typename matrix_traits<RArg>::value_type& c)
	{
		return mm_expr<LArg, RArg, LTag, RTag>(expr, expr.alpha() * c);
	}

	template<class LArg, class RArg, typename LTag, typename RTag>
	BCS_ENSURE_INLINE
	inline mm_expr<LArg, RArg, LTag, RTag> operator * (
			const typename matrix_traits<LArg>::value_type& c, const mm_expr<LArg, RArg, LTag, RTag>& expr)
	{
		return mm_expr<LArg, RArg, LTag, RTag>(expr, c * expr.alpha());
	}

	template<class DMat, class LArg, class RArg, typename LTag, typename RTag>
	BCS_ENSURE_INLINE
	inline void operator += (
			IDenseMatrix<DMat, typename matrix_traits<LArg>::value_type>& lhs,
			const mm_expr<LArg, RArg, LTag, RTag>& expr)
	{
		typedef typename matrix_traits<LArg>::value_type T;
		typedef typename mm_expr<LArg, RArg, LTag, RTag>::intern_t intern_t;

		matrix_capture<LArg, is_dense_mat<LArg>::value > left(expr.left_arg());
		matrix_capture<RArg, is_dense_mat<RArg>::value > right(expr.right_arg());

		intern_t::eval(expr.alpha(), left.get(), right.get(), T(1), lhs.derived());
	}


}

#endif /* MATRIX_PROD_H_ */
