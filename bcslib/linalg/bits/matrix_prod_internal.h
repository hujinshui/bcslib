/*
 * @file matrix_prod_internal.h
 *
 * Internal implementation of matrix product expression and dispatch
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PROD_INTERNAL_H_
#define BCSLIB_MATRIX_PROD_INTERNAL_H_

#include <bcslib/linalg/linalg_base.h>

namespace bcs { namespace detail {

	/********************************************
	 *
	 *  argument information
	 *
	 ********************************************/

	template<class LArg>
	struct mm_left_arg
	{
		typedef typename select_type<ct_is_row<LArg>::value,
				mm_row_tag, mm_mat_tag>::type tag;

		typedef LArg type;

		BCS_ENSURE_INLINE
		static const type& get(const LArg& in) { return in; }
	};

	template<class LArg>
	struct mm_left_arg<bcs::transpose_expr<LArg> >
	{
		typedef mm_tmat_tag tag;

		typedef LArg type;

		BCS_ENSURE_INLINE
		static const type& get(const bcs::transpose_expr<LArg>& in) { return in.arg(); }
	};


	template<class RArg>
	struct mm_right_arg
	{
		typedef typename select_type<ct_is_col<RArg>::value,
				mm_col_tag, mm_mat_tag>::type tag;

		typedef RArg type;

		BCS_ENSURE_INLINE
		static const type& get(const RArg& in) { return in; }
	};

	template<class RArg>
	struct mm_right_arg<bcs::transpose_expr<RArg> >
	{
		typedef mm_tmat_tag tag;

		typedef RArg type;

		BCS_ENSURE_INLINE
		static const type& get(const bcs::transpose_expr<RArg>& in) { return in.arg(); }
	};


	/********************************************
	 *
	 *  Internal implementation
	 *
	 ********************************************/

	template<class LArg, class RArg, typename LTag, typename RTag> struct mm_expr_intern;

	template<class Arg>
	struct mm_arg_capture
	{
		typedef typename matrix_capture<Arg, is_dense_mat<Arg>::value>::captured_type type;
	};

	// mat x col

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_mat_tag, mm_col_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.nrows();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.nrows(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return 1; }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemv_n(alpha, lmat, rmat, beta, dst);
		}
	};

	// tmat x col

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_tmat_tag, mm_col_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.nrows() == rarg.nrows(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.ncolumns();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.ncolumns(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return 1; }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemv_t(alpha, lmat, rmat, beta, dst);
		}
	};

	// row x mat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_row_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return rarg.ncolumns();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return 1; }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.ncolumns(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gevm_n(alpha, lmat, rmat, beta, dst);
		}
	};

	// row x tmat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_row_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.ncolumns() == rarg.ncolumns(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return rarg.nrows();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return 1; }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.nrows(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gevm_t(alpha, lmat, rmat, beta, dst);
		}
	};


	// mat x mat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_mat_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.nrows() * rarg.ncolumns();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.nrows(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.ncolumns(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemm_nn(alpha, lmat, rmat, beta, dst);
		}
	};

	// mat x tmat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_mat_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.ncolumns() == rarg.ncolumns(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.nrows() * rarg.rows();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.nrows(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.nrows(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemm_nt(alpha, lmat, rmat, beta, dst);
		}
	};

	// tmat x mat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_tmat_tag, mm_mat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.nrows() == rarg.nrows(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.ncolumns() * rarg.ncolumns();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.ncolumns(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.ncolumns(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemm_tn(alpha, lmat, rmat, beta, dst);
		}
	};

	// tmat x tmat

	template<class LArg, class RArg>
	struct mm_expr_intern<LArg, RArg, mm_tmat_tag, mm_tmat_tag>
	{
		typedef typename matrix_traits<LArg>::value_type T;

		BCS_ENSURE_INLINE static void check_args(const LArg& larg, const RArg& rarg)
		{
			check_arg(larg.nrows() == rarg.ncolumns(), "Inconsistent inner dimension for mm_expr");
		}

		BCS_ENSURE_INLINE static index_t get_nelems(const LArg& larg, const RArg& rarg)
		{
			return larg.ncolumns() * rarg.nrows();
		}

		BCS_ENSURE_INLINE static index_t get_nrows(const LArg& larg) { return larg.ncolumns(); }

		BCS_ENSURE_INLINE static index_t get_ncols(const RArg& rarg) { return rarg.nrows(); }

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha,
				const typename mm_arg_capture<LArg>::type& lmat,
				const typename mm_arg_capture<RArg>::type& rmat,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			blas::gemm_tt(alpha, lmat, rmat, beta, dst);
		}
	};



} }

#endif 
