/**
 * @file column_traverser.h
 *
 * The class for traversing across columns
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_COLUMN_TRAVERSER_H_
#define BCSLIB_COLUMN_TRAVERSER_H_

#include <bcslib/matrix/dense_matrix.h>

namespace bcs
{

	namespace detail
	{
		struct coltrav_dense_tag { };
		struct coltrav_view_tag { };
		struct coltrav_obscure_tag { };

		template<class Expr, typename ColTravKind> struct column_traverser_impl;

		template<class Expr>
		struct column_traverser_impl<Expr, coltrav_dense_tag>
		{
#ifdef BCS_USE_STATIC_ASSERT
			static_assert(has_matrix_interface<Expr, IDenseMatrix>::value,
					"Expr must be a model of IDenseMatrix");
#endif

			typedef typename matrix_traits<Expr>::value_type value_type;

			const value_type *m_ptr;
			const index_t m_stride;

			BCS_ENSURE_INLINE
			column_traverser_impl(const Expr& mat)
			: m_ptr(mat.ptr_data()), m_stride(mat.lead_dim()) { }

			BCS_ENSURE_INLINE
			value_type get(index_t i) const { return m_ptr[i]; }

			BCS_ENSURE_INLINE
			void move_next() { m_ptr += m_stride; }

			BCS_ENSURE_INLINE
			void move_prev() { m_ptr -= m_stride; }

			BCS_ENSURE_INLINE
			void move(index_t step) { m_ptr += m_stride * step; }
		};


		template<class Expr>
		struct column_traverser_impl<Expr, coltrav_view_tag>
		{
#ifdef BCS_USE_STATIC_ASSERT
			static_assert(has_matrix_interface<Expr, IMatrixView>::value,
					"Expr must be a model of IMatrixView");

			static_assert(!(has_matrix_interface<Expr, IDenseMatrix>::value),
					"Expr must NOT be a model of IDenseMatrix");
#endif

			typedef typename matrix_traits<Expr>::value_type value_type;

			const Expr& m_mat;
			const index_t m_icol;

			BCS_ENSURE_INLINE
			column_traverser_impl(const Expr& mat)
			: m_mat(mat), m_icol(0) { }

			BCS_ENSURE_INLINE
			value_type get(index_t i) const { return m_mat.elem(i, m_icol); }

			BCS_ENSURE_INLINE
			void move_next() { ++ m_icol; }

			BCS_ENSURE_INLINE
			void move_prev() { -- m_icol; }

			BCS_ENSURE_INLINE
			void move(index_t step) { m_icol += step; }
		};


		template<class Expr>
		struct column_traverser_impl<Expr, coltrav_obscure_tag>
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
			column_traverser_impl(const Expr& expr)
			: m_mat(expr), m_ptr(m_mat.ptr_data()), m_stride(m_mat.lead_dim())
			{ }

			BCS_ENSURE_INLINE
			value_type get(index_t i) const { return m_ptr[i]; }

			BCS_ENSURE_INLINE
			void move_next() { m_ptr += m_stride; }

			BCS_ENSURE_INLINE
			void move_prev() { m_ptr -= m_stride; }

			BCS_ENSURE_INLINE
			void move(index_t step) { m_ptr += m_stride * step; }
		};

	}


	template<class Expr>
	class column_traverser
	{
		typedef typename
				select_type<
					has_matrix_interface<Expr, IMatrixView>::value,
					typename
					select_type<
						has_matrix_interface<Expr, IDenseMatrix>::value,
						detail::coltrav_dense_tag,
						detail::coltrav_view_tag>::type,
					detail::coltrav_obscure_tag>::type coltrav_tag;

		typedef detail::column_traverser_impl<Expr, coltrav_tag> internal_t;

	public:
		typedef typename matrix_traits<Expr>::value_type value_type;

		BCS_ENSURE_INLINE
		column_traverser(const Expr& mat) : m_internal(mat) { }

		value_type operator[] (index_t i) const
		{
			return m_internal.get(i);
		}

		BCS_ENSURE_INLINE column_traverser& operator ++ ()
		{
			m_internal.move_next();
			return *this;
		}

		BCS_ENSURE_INLINE column_traverser& operator -- ()
		{
			m_internal.move_prev();
			return *this;
		}

		BCS_ENSURE_INLINE column_traverser& operator += (index_t n)
		{
			m_internal.move(n);
			return *this;
		}

		BCS_ENSURE_INLINE column_traverser& operator -= (index_t n)
		{
			m_internal.move(-n);
			return *this;
		}

	private:
		internal_t m_internal;
	};

}

#endif /* COLUMN_TRAVERSER_H_ */
