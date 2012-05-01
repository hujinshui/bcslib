/**
 * @file matrix_transpose.h
 *
 * Expression to represent matrix/vector transposition
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_TRANSPOSE_H_
#define BCSLIB_MATRIX_TRANSPOSE_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/bits/vector_transpose_internal.h>
#include <bcslib/matrix/bits/matrix_transpose_internal.h>

#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/ref_matrix.h>

namespace bcs
{

	// forward declaration

	template<class Arg> class transposed_view_proxy;
	template<class Arg> class transposed_dense_proxy;

	template<class Mat> class matrix_transpose_expr;


	/********************************************
	 *
	 *  Vector transpose proxy
	 *
	 ********************************************/

	template<class Arg>
	struct matrix_traits<transposed_dense_proxy<Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<Arg>::value;
		static const int compile_time_num_cols = ct_rows<Arg>::value;

		static const bool is_linear_indexable = bcs::matrix_traits<Arg>::is_linear_indexable;
		static const bool is_continuous = bcs::matrix_traits<Arg>::is_continuous || ct_cols<Arg>::value == 1;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename bcs::matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg>
	class transposed_dense_proxy
	: public IDenseMatrix<transposed_dense_proxy<Arg>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IDenseMatrix>,
				"Arg must be a model of IDenseMatrix");
#endif

		static const bool IsCompileTimeVector = ct_cols<Arg>::value == 1 || ct_rows<Arg>::value == 1;

	public:
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		BCS_ENSURE_INLINE
		explicit transposed_dense_proxy(const Arg& arg)
		: m_arg(arg)
		{
			if (!IsCompileTimeVector)
			{
				check_arg(bcs::is_vector(arg),
					"The argument to transposed_dense_proxy must be a vector.");
			}

			m_transposed_leaddim = (bcs::is_column(arg) ? 1 : arg.nrows());
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const { return m_arg.nelems(); }

		BCS_ENSURE_INLINE size_type size() const { return m_arg.size(); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_arg.ncolumns(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_arg.nrows(); }

		BCS_ENSURE_INLINE const_pointer ptr_data() const { return m_arg.ptr_data(); }

		BCS_ENSURE_INLINE index_type lead_dim() const { return m_transposed_leaddim; }

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_arg.elem(j, i);
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return get_by_linear_index(m_arg, i);
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			throw invalid_operation(
					"Attempted to resize a transposed proxy, which is not allowed.");
		}

	private:
		const Arg& m_arg;
		const index_t m_transposed_leaddim;
	};


	template<class Arg>
	struct matrix_traits<transposed_view_proxy<Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<Arg>::value;
		static const int compile_time_num_cols = ct_rows<Arg>::value;

		static const bool is_linear_indexable = bcs::matrix_traits<Arg>::is_linear_indexable;
		static const bool is_continuous = bcs::matrix_traits<Arg>::is_continuous || ct_cols<Arg>::value == 1;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename bcs::matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg>
	class transposed_view_proxy
	: public IDenseMatrix<transposed_view_proxy<Arg>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IMatrixView>,
				"Arg must be a model of IMatrixView");
#endif

		static const bool IsCompileTimeVector = ct_cols<Arg>::value == 1 || ct_rows<Arg>::value == 1;

	public:
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		BCS_ENSURE_INLINE
		explicit transposed_view_proxy(const Arg& arg)
		: m_arg(arg)
		{
			if (!IsCompileTimeVector)
			{
				check_arg(bcs::is_vector(arg),
					"The argument to transposed_dense_proxy must be a vector.");
			}
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const { return m_arg.nelems(); }

		BCS_ENSURE_INLINE size_type size() const { return m_arg.size(); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_arg.ncolumns(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_arg.nrows(); }

		BCS_ENSURE_INLINE value_type elem(index_type i, index_type j) const
		{
			return m_arg.elem(j, i);
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return get_by_linear_index(m_arg, i);
		}

	private:
		const Arg& m_arg;
	};


	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	inline transposed_dense_proxy<Mat> transpose_vec(const IDenseMatrix<Mat, T>& vec)
	{
		return transposed_dense_proxy<Mat>(vec.derived());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	inline transposed_view_proxy<Mat> transpose_vec(const IMatrixView<Mat, T>& vec)
	{
		return transposed_view_proxy<Mat>(vec.derived());
	}


	template<class Arg>
	struct expr_evaluator<transposed_dense_proxy<Arg> >
	{
		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef transposed_dense_proxy<Arg> SExpr;

		template<class DMat>
		inline static void evaluate(const SExpr& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			typedef typename detail::vec_trans_tag<SExpr>::type STag;
			typedef typename detail::vec_trans_tag<DMat>::type DTag;

			detail::vec_trans_evaluator<SExpr, DMat, STag, DTag>::run(expr, dst.derived());
		}
	};



	/********************************************
	 *
	 *  Generic Matrix transpose expression
	 *
	 ********************************************/

	template<class Arg>
	struct matrix_traits<matrix_transpose_expr<Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<Arg>::value;
		static const int compile_time_num_cols = ct_rows<Arg>::value;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg>
	struct matrix_transpose_expr
	: public IMatrixXpr<matrix_transpose_expr<Arg>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Arg arg_type;
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		const Arg& arg;

		matrix_transpose_expr(const Arg& a)
		: arg(a)
		{
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return arg.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return arg.ncolumns();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return arg.nrows();
		}
	};


	template<class Arg>
	struct expr_evaluator<matrix_transpose_expr<Arg> >
	{
		typedef matrix_transpose_expr<Arg> SExpr;
		typedef typename matrix_traits<SExpr>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const SExpr& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::matrix_transposer<SExpr, DMat,
				bcs::has_matrix_interface<DMat, IDenseMatrix>::value>::run(expr, dst.derived());
		}
	};



	/********************************************
	 *
	 *  Compile-time dispatch
	 *
	 ********************************************/


	namespace detail
	{

		struct transpose_generic_tag { };
		struct transpose_vecview_tag { };
		struct transpose_densevec_tag { };

		template<class Arg>
		struct transpose_tag
		{
			typedef typename select_type<
					has_matrix_interface<Arg, IMatrixView>::value &&
					(ct_rows<Arg>::value == 1 || ct_cols<Arg>::value == 1),
						typename select_type<
						has_matrix_interface<Arg, IDenseMatrix>::value,
							transpose_densevec_tag,
							transpose_vecview_tag
						>::type,
						transpose_generic_tag
					>::type type;
		};

		template<class Arg, bool IsCTVec> struct matrix_transpose_dispatch_helper;

		template<class Arg>
		struct matrix_transpose_dispatch_helper<Arg, transpose_generic_tag>
		{
			typedef matrix_transpose_expr<Arg> result_type;
		};

		template<class Arg>
		struct matrix_transpose_dispatch_helper<Arg, transpose_densevec_tag>
		{
			typedef transposed_dense_proxy<Arg> result_type;
		};

		template<class Arg>
		struct matrix_transpose_dispatch_helper<Arg, transpose_vecview_tag>
		{
			typedef transposed_view_proxy<Arg> result_type;
		};
	}


	template<class Arg>
	struct matrix_transpose_dispatcher
	{
		typedef typename detail::transpose_tag<Arg>::type tag_type;
		typedef typename detail::matrix_transpose_dispatch_helper<Arg, tag_type>::result_type result_type;

		BCS_ENSURE_INLINE
		static result_type run(const Arg& arg)
		{
			return result_type(arg);
		}
	};

	template<class Arg, typename T>
	BCS_ENSURE_INLINE
	typename matrix_transpose_dispatcher<Arg>::result_type
	transpose(const IMatrixXpr<Arg, T>& arg)
	{
		return matrix_transpose_dispatcher<Arg>::run(arg.derived());
	}
}

#endif /* MATRIX_TRANSPOSE_H_ */
