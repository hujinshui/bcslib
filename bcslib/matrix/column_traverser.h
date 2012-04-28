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

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{

	template<class Expr>
	struct is_column_traversable
	{
		static const bool value = has_matrix_interface<Expr, IMatrixView>::value;
	};


	namespace detail
	{
		template<class Expr, bool IsDense> struct column_traverser_impl;

		template<class Expr>
		struct column_traverser_impl<Expr, true>
		{
			typedef typename matrix_traits<Expr>::value_type value_type;

			const value_type *ptr;
			const index_t stride;

			BCS_ENSURE_INLINE
			column_traverser_impl(const Expr& mat)
			: ptr(mat.ptr_data()), stride(mat.lead_dim()) { }

			BCS_ENSURE_INLINE
			value_type get(index_t i) const { return ptr[i]; }

			BCS_ENSURE_INLINE
			void move_next() { ptr += stride; }

			BCS_ENSURE_INLINE
			void move_prev() { ptr -= stride; }

			BCS_ENSURE_INLINE
			void move(index_t step) { ptr += stride * step; }
		};


		template<class Expr>
		struct column_traverser_impl<Expr, false>
		{
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

	}


	template<class Expr>
	class column_traverser
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Expr, IMatrixView>::value,
				"Expr must be a model of IMatrixView");
#endif

		typedef detail::column_traverser_impl<Expr,
				has_matrix_interface<Expr, IDenseMatrix>::value> internal_t;

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
