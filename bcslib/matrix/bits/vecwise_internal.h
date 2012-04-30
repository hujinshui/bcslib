/**
 * @file vecwise_internal.h
 *
 * Internal implementation for vector-wise traversal and calculation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef VECWISE_INTERNAL_H_
#define VECWISE_INTERNAL_H_

#include <bcslib/matrix/vector_proxy_base.h>
#include <bcslib/matrix/dense_matrix.h>

namespace bcs { namespace detail {


	struct const_vecwise_dense_tag { };
	struct const_vecwise_view_tag { };
	struct const_vecwise_obscure_tag { };

	struct vecwise_dense_tag { };
	struct vecwise_regular_tag { };

	template<class Expr, typename VecKind> struct vecwise_reader_impl;
	template<class Expr, typename VecKind> struct vecwise_writer_impl;


	template<class Expr>
	struct vecwise_reader_impl<Expr, const_vecwise_dense_tag>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IDenseMatrix>::value,
				"Expr must be a model of IDenseMatrix");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		const value_type *m_ptr;
		const index_t m_stride;

		BCS_ENSURE_INLINE
		vecwise_reader_impl(const Expr& mat)
		: m_ptr(mat.ptr_data()),  m_stride(mat.lead_dim())
		{ }

		BCS_ENSURE_INLINE
		value_type load_scalar(index_t i) const { return m_ptr[i]; }

		BCS_ENSURE_INLINE
		void move_next() { m_ptr += m_stride; }

		BCS_ENSURE_INLINE
		void move_prev() { m_ptr -= m_stride; }

		BCS_ENSURE_INLINE
		void move(index_t step) { m_ptr += m_stride * step; }
	};


	template<class Expr>
	struct vecwise_reader_impl<Expr, const_vecwise_view_tag>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IMatrixView>::value,
				"Expr must be a model of IMatrixView");

		static_assert(!(has_matrix_interface<Expr, IDenseMatrix>::value),
				"Expr must NOT be a model of IDenseMatrix");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		const Expr& m_mat;
		index_t m_icol;

		BCS_ENSURE_INLINE
		vecwise_reader_impl(const Expr& mat)
		: m_mat(mat), m_icol(0) { }

		BCS_ENSURE_INLINE
		value_type load_scalar(index_t i) const { return m_mat.elem(i, m_icol); }

		BCS_ENSURE_INLINE
		void move_next() { ++ m_icol; }

		BCS_ENSURE_INLINE
		void move_prev() { -- m_icol; }

		BCS_ENSURE_INLINE
		void move(index_t step) { m_icol += step; }
	};


	template<class Expr>
	struct vecwise_reader_impl<Expr, const_vecwise_obscure_tag>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IMatrixXpr>::value,
				"Expr must be a model of IMatrixXpr");
		static_assert(!(has_matrix_interface<Expr, IMatrixView>::value),
				"Expr must NOT be a model of IMatrixView");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		typedef dense_matrix<
				typename matrix_traits<Expr>::value_type,
				ct_rows<Expr>::value,
				ct_cols<Expr>::value> internal_matrix_t;

		internal_matrix_t m_mat;
		const value_type* m_ptr;
		const index_t m_stride;

		BCS_ENSURE_INLINE
		vecwise_reader_impl(const Expr& expr)
		: m_mat(expr)
		, m_ptr(m_mat.ptr_data())
		, m_stride(m_mat.lead_dim())
		{ }

		BCS_ENSURE_INLINE
		value_type load_scalar(index_t i) const { return m_ptr[i]; }

		BCS_ENSURE_INLINE
		void move_next() { m_ptr += m_stride; }

		BCS_ENSURE_INLINE
		void move_prev() { m_ptr -= m_stride; }

		BCS_ENSURE_INLINE
		void move(index_t step) { m_ptr += m_stride * step; }
	};


	template<class Expr>
	struct vecwise_writer_impl<Expr, vecwise_dense_tag>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IDenseMatrix>::value,
				"Expr must be a model of IDenseMatrix");

		static_assert(!(matrix_traits<Expr>::is_readonly), "Expr must NOT be read-only.");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		value_type *m_ptr;
		const index_t m_stride;

		BCS_ENSURE_INLINE
		vecwise_writer_impl(Expr& mat)
		: m_ptr(mat.ptr_data()), m_stride(mat.lead_dim()) { }

		BCS_ENSURE_INLINE
		void store_scalar(index_t i, const value_type& v) { m_ptr[i] = v; }

		BCS_ENSURE_INLINE
		void move_next() { m_ptr += m_stride; }

		BCS_ENSURE_INLINE
		void move_prev() { m_ptr -= m_stride; }

		BCS_ENSURE_INLINE
		void move(index_t step) { m_ptr += m_stride * step; }
	};


	template<class Expr>
	struct vecwise_writer_impl<Expr, vecwise_regular_tag>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IMatrixView>::value,
				"Expr must be a model of IMatrixView");

		static_assert(!(has_matrix_interface<Expr, IDenseMatrix>::value),
				"Expr must NOT be a model of IDenseMatrix");

		static_assert(!(matrix_traits<Expr>::is_readonly), "Expr must NOT be read-only.");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		Expr& m_mat;
		index_t m_icol;

		BCS_ENSURE_INLINE
		vecwise_writer_impl(Expr& mat)
		: m_mat(mat), m_icol(0) { }

		BCS_ENSURE_INLINE
		void store_scalar(index_t i, const value_type& v) const { m_mat.elem(i, m_icol) = v; }

		BCS_ENSURE_INLINE
		void move_next() { ++ m_icol; }

		BCS_ENSURE_INLINE
		void move_prev() { -- m_icol; }

		BCS_ENSURE_INLINE
		void move(index_t step) { m_icol += step; }
	};

} }




#endif /* VECWISE_INTERNAL_H_ */
