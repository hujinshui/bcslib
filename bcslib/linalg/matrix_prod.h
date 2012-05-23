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

#include <bcslib/linalg/mm_evaluators.h>

namespace bcs
{

	/********************************************
	 *
	 *  Expression classes
	 *
	 ********************************************/

	template<class LArg, class RArg> class mm_expr;

	template<class LArg, class RArg>
	struct matrix_traits<mm_expr<LArg, RArg> >
	{
		static const int num_dimensions = 2;

		static const int compile_time_num_rows = ct_rows<LArg>::value;
		static const int compile_time_num_cols = ct_cols<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};

	template<class LArg, class RArg>
	class mm_expr : public IMatrixXpr<mm_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<LArg>::value, "LArg must be a matrix expression");
		static_assert(is_mat_xpr<RArg>::value, "LArg must be a matrix expression");
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value, "LArg and RArg must have the same type.");
#endif

	public:
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

		BCS_ENSURE_INLINE
		mm_expr(const LArg& larg, const RArg& rarg, const value_type alpha = value_type(1))
		: m_left_arg(larg), m_right_arg(rarg), m_alpha(alpha)
		{
			check_arg(larg.ncolumns() == rarg.nrows(),
					"Inner dimensions are inconsistent (for mm_expr)");
		}

		BCS_ENSURE_INLINE
		mm_expr(const mm_expr& e0, const value_type alpha)  // replace the alpha value
		: m_left_arg(e0.m_left_arg), m_right_arg(e0.m_right_arg), m_alpha(alpha)
		{
		}

		BCS_ENSURE_INLINE
		const LArg& left_arg() const
		{
			return m_left_arg;
		}

		BCS_ENSURE_INLINE
		const RArg& right_arg() const
		{
			return m_right_arg;
		}

		BCS_ENSURE_INLINE
		value_type alpha() const
		{
			return m_alpha;
		}

		BCS_ENSURE_INLINE
		index_type nelems() const
		{
			return nrows() * ncolumns();
		}

		BCS_ENSURE_INLINE
		size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE
		index_type nrows() const
		{
			return m_left_arg.nrows();
		}

		BCS_ENSURE_INLINE
		index_type ncolumns() const
		{
			return m_right_arg.ncolumns();
		}

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
		const value_type m_alpha;
	};


	/********************************************
	 *
	 *  Evaluators
	 *
	 ********************************************/

	template<class LArg, class RArg>
	struct expr_evaluator<mm_expr<LArg, RArg> >
	{
		typedef mm_expr<LArg, RArg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			mm_evaluator<LArg, RArg>::eval(
					expr.alpha(), expr.left_arg(), expr.right_arg(), T(0), dst);
		}
	};


	/********************************************
	 *
	 *  Expression construction
	 *
	 ********************************************/

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline mm_expr<LArg, RArg> mm(const IMatrixXpr<LArg, T>& a, const IMatrixXpr<RArg, T>& b)
	{
		return mm_expr<LArg, RArg>(a.derived(), b.derived());
	}


	template<class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline mm_expr<LArg, RArg> operator * (
			const mm_expr<LArg, RArg>& expr, const typename matrix_traits<RArg>::value_type& c)
	{
		return mm_expr<LArg, RArg>(expr, expr.alpha() * c);
	}

	template<class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline mm_expr<LArg, RArg> operator * (
			const typename matrix_traits<LArg>::value_type& c, const mm_expr<LArg, RArg>& expr)
	{
		return mm_expr<LArg, RArg>(expr, c * expr.alpha());
	}

	template<class DMat, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline void operator += (
			IDenseMatrix<DMat, typename matrix_traits<LArg>::value_type>& lhs,
			const mm_expr<LArg, RArg>& expr)
	{
		typedef typename matrix_traits<LArg>::value_type T;

		mm_evaluator<LArg, RArg>::eval(
				expr.alpha(), expr.left_arg(), expr.right_arg(), T(1), lhs);
	}


}

#endif /* MATRIX_PROD_H_ */
