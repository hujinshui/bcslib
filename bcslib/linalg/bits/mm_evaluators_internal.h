/**
 * @file mm_evaluators_internal.h
 *
 * Internal implementation for mm_evaluators
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MM_EVALUATORS_INTERNAL_H_
#define BCSLIB_MM_EVALUATORS_INTERNAL_H_

#include <bcslib/linalg/linalg_base.h>
#include <bcslib/matrix/matrix_capture.h>

namespace bcs { namespace detail {

	struct mm_row_tag { };
	struct mm_col_tag { };
	struct mm_mat_tag { };
	struct mm_tmat_tag { };
	struct mm_smat_tag { };

	template<class LArg, class RArg, typename LTag, typename RTag> struct mm_evaluator_intern;

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_mat_tag, mm_col_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::gemv_n(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_tmat_tag, mm_col_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<typename LArg::arg_type, is_dense_mat<typename LArg::arg_type>::value> left(larg.arg());
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::gemv_t(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_smat_tag, mm_col_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<typename LArg::mat_type, is_dense_mat<typename LArg::mat_type>::value> left(larg.get());
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::symv(alpha, left.get(), right.get(), beta, dst);
		}
	};


	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_row_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::gevm_n(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_row_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<typename RArg::arg_type, is_dense_mat<typename RArg::arg_type>::value> right(rarg.arg());

			blas::gevm_t(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_row_tag, mm_smat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<typename RArg::mat_type, is_dense_mat<typename RArg::mat_type>::value> right(rarg.get());

			blas::syvm(alpha, left.get(), right.get(), beta, dst);
		}
	};




	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_mat_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::gemm_nn(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_mat_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<typename RArg::arg_type, is_dense_mat<typename RArg::arg_type>::value> right(rarg.arg());

			blas::gemm_nt(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_mat_tag, mm_smat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg>::value> left(larg);
			matrix_capture<typename RArg::mat_type, is_dense_mat<typename RArg::mat_type>::value> right(rarg.get());

			blas::symm_r(alpha, left.get(), right.get(), beta, dst);
		}
	};


	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_tmat_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<typename LArg::arg_type, is_dense_mat<typename LArg::arg_type>::value> left(larg.arg());
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::gemm_tn(alpha, left.get(), right.get(), beta, dst);
		}
	};

	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_tmat_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<typename LArg::arg_type, is_dense_mat<typename LArg::arg_type>::value> left(larg.arg());
			matrix_capture<typename RArg::arg_type, is_dense_mat<typename RArg::arg_type>::value> right(rarg.arg());

			blas::gemm_tt(alpha, left.get(), right.get(), beta, dst);
		}
	};


	template<class LArg, class RArg>
	struct mm_evaluator_intern<LArg, RArg, mm_smat_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<typename LArg::mat_type, is_dense_mat<typename LArg::mat_type>::value> left(larg.get());
			matrix_capture<RArg, is_dense_mat<RArg>::value> right(rarg);

			blas::symm_l(alpha, left.get(), right.get(), beta, dst);
		}
	};



	template<class LArg> struct mm_left_tag
	{
		typedef typename select_type<ct_is_row<LArg>::value,
				mm_row_tag, mm_mat_tag>::type type;
	};

	template<class LArg> struct mm_left_tag<transpose_expr<LArg> > { typedef mm_tmat_tag type; };

	template<class LArg> struct mm_left_tag<sym_mat_proxy<LArg> > { typedef mm_smat_tag type; };

	template<class RArg> struct mm_right_tag
	{
		typedef typename select_type<ct_is_col<RArg>::value,
				mm_col_tag, mm_mat_tag>::type type;
	};

	template<class RArg> struct mm_right_tag<transpose_expr<RArg> > { typedef mm_tmat_tag type; };

	template<class RArg> struct mm_right_tag<sym_mat_proxy<RArg> > { typedef mm_smat_tag type; };


	template<class LArg, class RArg>
	struct mm_evaluator_dispatcher
	{
		typedef typename mm_left_tag<LArg>::type left_tag;
		typedef typename mm_right_tag<RArg>::type right_tag;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert( !( bcs::is_same<left_tag, mm_smat_tag>::value && bcs::is_same<right_tag, mm_tmat_tag>::value ),
				"It is not allowed to use sym_mat_proxy and transpose_expr as two operands of mm.");

		static_assert( !( bcs::is_same<left_tag, mm_tmat_tag>::value && bcs::is_same<right_tag, mm_smat_tag>::value ),
				"It is not allowed to use transpose_expr and sym_mat_expr as two operands of mm.");

		static_assert( !( bcs::is_same<left_tag, mm_smat_tag>::value && bcs::is_same<right_tag, mm_smat_tag>::value ),
				"It is not allowed to use sym_mat_proxy as both operands of mm.");
#endif

		typedef mm_evaluator_intern<LArg, RArg, left_tag, right_tag> type;
	};


} }

#endif /* MM_EVALUATORS_INTERNAL_H_ */
