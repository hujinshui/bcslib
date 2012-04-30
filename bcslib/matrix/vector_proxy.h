/**
 * @file vector_proxy.h
 *
 * A special class to support vector-based traversal and calculation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_COLUMN_TRAVERSER_H_
#define BCSLIB_COLUMN_TRAVERSER_H_

#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/bits/vecwise_internal.h>

namespace bcs
{
	// forward declaration

	template<class Expr> class vec_reader;

	template<class Expr> class vecwise_reader;
	template<class Expr> class vecwise_writer;


	/********************************************
	 *
	 *   single-vector read/write
	 *
	 ********************************************/

	template<class Expr>
	struct is_accessible_as_vector
	{
		static const bool value = matrix_traits<Expr>::is_linear_indexable;
	};


	template<class Expr>
	class vec_reader
	: public IVecReader<vec_reader<Expr>, typename matrix_traits<Expr>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(matrix_traits<Expr>::is_linear_indexable, "Expr must be linearly indexable");
#endif

	public:
		typedef typename matrix_traits<Expr>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vec_reader(const Expr& expr) : m_vec(expr) { }

		BCS_ENSURE_INLINE
		value_type load_scalar(index_t i) const
		{
			return m_vec[i];
		}

	private:
		const Expr& m_vec;
	};


	template<class Expr>
	class vec_writer
	: public IVecWriter<vec_writer<Expr>, typename matrix_traits<Expr>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(matrix_traits<Expr>::is_linear_indexable, "Expr must be linearly indexable");
		static_assert((!matrix_traits<Expr>::is_readonly), "Expr must NOT be readonly");
#endif

	public:
		typedef typename matrix_traits<Expr>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vec_writer(Expr& expr) : m_vec(expr) { }

		BCS_ENSURE_INLINE
		void store_scalar(index_t i, const value_type& v)
		{
			m_vec[i] = v;
		}

	private:
		Expr& m_vec;
	};



	/********************************************
	 *
	 *   vecwise read/write
	 *
	 ********************************************/

	template<class Expr>
	class vecwise_reader
	: public IVecReader<vecwise_reader<Expr>, typename matrix_traits<Expr>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::has_matrix_interface<Expr, IMatrixXpr>::value,
				"Expr must be a model of IMatrixXpr");
#endif

		typedef typename
				select_type<
					has_matrix_interface<Expr, IMatrixView>::value,
					typename
					select_type<
						has_matrix_interface<Expr, IDenseMatrix>::value,
						detail::const_vecwise_dense_tag,
						detail::const_vecwise_view_tag>::type,
					detail::const_vecwise_obscure_tag>::type vec_kind_t;

		typedef detail::vecwise_reader_impl<Expr, vec_kind_t> internal_t;

	public:
		typedef typename matrix_traits<Expr>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vecwise_reader(const Expr& mat) : m_internal(mat) { }

		BCS_ENSURE_INLINE value_type load_scalar(index_t i) const
		{
			return m_internal.load_scalar(i);
		}

		BCS_ENSURE_INLINE void operator ++ ()
		{
			m_internal.move_next();
		}

		BCS_ENSURE_INLINE void operator -- ()
		{
			m_internal.move_prev();
		}

		BCS_ENSURE_INLINE void operator += (index_t n)
		{
			m_internal.move(n);
		}

		BCS_ENSURE_INLINE void operator -= (index_t n)
		{
			m_internal.move(-n);
		}

	private:
		internal_t m_internal;
	};


	template<class Expr>
	class vecwise_writer
	: public IVecWriter<vecwise_writer<Expr>, typename matrix_traits<Expr>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::has_matrix_interface<Expr, IRegularMatrix>::value,
				"Expr must be a model of IMatrixXpr");

		static_assert(!(matrix_traits<Expr>::is_readonly), "Expr must NOT be read-only.");
#endif

		typedef typename
				select_type<
					has_matrix_interface<Expr, IDenseMatrix>::value,
						detail::vecwise_dense_tag,
						detail::vecwise_regular_tag>::type vec_kind_t;

		typedef detail::vecwise_writer_impl<Expr, vec_kind_t> internal_t;

	public:
		typedef typename matrix_traits<Expr>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit vecwise_writer(Expr& mat) : m_internal(mat) { }

		BCS_ENSURE_INLINE void store_scalar(index_t i, const value_type& v)
		{
			m_internal.store_scalar(i, v);
		}

		BCS_ENSURE_INLINE void operator ++ ()
		{
			m_internal.move_next();
		}

		BCS_ENSURE_INLINE void operator -- ()
		{
			m_internal.move_prev();
		}

		BCS_ENSURE_INLINE void operator += (index_t n)
		{
			m_internal.move(n);
		}

		BCS_ENSURE_INLINE void operator -= (index_t n)
		{
			m_internal.move(-n);
		}

	private:
		internal_t m_internal;
	};

}

#endif /* COLUMN_TRAVERSER_H_ */
