/**
 * @file gen_matrix_prod.h
 *
 * General Matrix product expression and evaluation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_GEN_MATRIX_PROD_H_
#define BCSLIB_GEN_MATRIX_PROD_H_

#include <bcslib/linalg/matrix_blas.h>
#include <bcslib/matrix/matrix_capture.h>

namespace bcs
{

	// forward declarations

	template<class LArg, class RArg> class gemv_n_expr;
	template<class LArg, class RArg> class gemv_t_expr;
	template<class LArg, class RArg> class gevm_n_expr;
	template<class LArg, class RArg> class gevm_t_expr;

	template<class LArg, class RArg> class gemm_nn_expr;
	template<class LArg, class RArg> class gemm_nt_expr;
	template<class LArg, class RArg> class gemm_tn_expr;
	template<class LArg, class RArg> class gemm_tt_expr;


	/********************************************
	 *
	 *  Expression classes
	 *
	 ********************************************/

	// gemv_n_expr

	template<class LArg, class RArg>
	struct matrix_traits<gemv_n_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<LArg>::value;
		static const int compile_time_num_cols = 1;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gemv_n_expr
	: public IMatrixXpr<gemv_n_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

	public:
		gemv_n_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE LArg& left_arg() const { return m_left_arg; }



		BCS_ENSURE_INLINE index_type nelems() const { return nrows(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.nrows(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return 1; }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gemv_t_expr

	template<class LArg, class RArg>
	struct matrix_traits<gemv_t_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<LArg>::value;
		static const int compile_time_num_cols = 1;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gemv_t_expr
	: public IMatrixXpr<gemv_t_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

	public:
		gemv_t_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.rows() == rarg.nrows(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return nrows(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.ncolumns(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return 1; }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gevm_n_expr

	template<class LArg, class RArg>
	struct matrix_traits<gevm_n_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = 1;
		static const int compile_time_num_cols = ct_cols<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gevm_n_expr
	: public IMatrixXpr<gevm_n_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

	public:
		gevm_n_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return 1; }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.ncolumns(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gevm_t_expr

	template<class LArg, class RArg>
	struct matrix_traits<gevm_t_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = 1;
		static const int compile_time_num_cols = ct_rows<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gevm_t_expr
	: public IMatrixXpr<gevm_t_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(T)

	public:
		gevm_t_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.ncolumns() == rarg.ncolumns(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return 1; }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.nrows(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};



	// gemm_nn

	template<class LArg, class RArg>
	struct matrix_traits<gemm_nn_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<LArg>::value;
		static const int compile_time_num_cols = ct_cols<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gemm_nn_expr
	: public IMatrixXpr<gemm_nn_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

	public:
		gemm_nn_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.ncolumns() == rarg.nrows(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return nrows() * ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.nrows(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.ncolumns(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gemm_nt

	template<class LArg, class RArg>
	struct matrix_traits<gemm_nt_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<LArg>::value;
		static const int compile_time_num_cols = ct_rows<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<typename T, class LArg, class RArg>
	class gemm_nt_expr
	: public IMatrixXpr<gemm_nt_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(T)

	public:
		gemm_nt_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.ncolumns() == rarg.ncolumns(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return nrows() * ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.nrows(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.nrows(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gemm_tn

	template<class LArg, class RArg>
	struct matrix_traits<gemm_tn_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<LArg>::value;
		static const int compile_time_num_cols = ct_cols<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gemm_tn_expr
	: public IMatrixXpr<gemm_tn_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(T)

	public:
		gemm_tn_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.nrows() == rarg.nrows(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return nrows() * ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.ncolumns(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.ncolumns(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	// gemm_tt

	template<class LArg, class RArg>
	struct matrix_traits<gemm_tt_expr<LArg, RArg> >
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<
				typename matrix_traits<LArg>::value_type,
				typename matrix_traits<RArg>::value_type>::value,
				"LArg and RArg should have the same value_type");
#endif

		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<LArg>::value;
		static const int compile_time_num_cols = ct_rows<RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<LArg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class LArg, class RArg>
	class gemm_tt_expr
	: public IMatrixXpr<gemm_tt_expr<LArg, RArg>, typename matrix_traits<LArg>::value_type>
	{
	public:
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<LArg>::value_type)

	public:
		gemm_tt_expr(const LArg& larg, const RArg& rarg)
		: m_left_arg(larg), m_right_arg(rarg)
		{
			check_arg(larg.nrows() == rarg.ncolumns(),
					"The inner dimension is not consistent (for gemm_nn_expr).");
		}

		BCS_ENSURE_INLINE index_type nelems() const { return nrows() * ncolumns(); }

		BCS_ENSURE_INLINE size_type size() const { return static_cast<size_type>(nelems()); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_left_arg.ncolumns(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_right_arg.nrows(); }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
	};


	/********************************************
	 *
	 *  Evaluator classes
	 *
	 ********************************************/

	template<class LArg, class RArg>
	struct expr_evaluator<gemv_n_expr<LArg, RArg> >
	{
		typedef gemv_n_expr<LArg, RArg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			matrix_capture<LArg, is_dense_mat<LArg> > left(expr.)

			blas::gemv_n()
		}
	};



}

#endif /* MATRIX_PROD_H_ */
