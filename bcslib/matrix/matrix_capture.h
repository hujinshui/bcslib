/*
 * @file matrix_capture.h
 *
 * A class to capture matrix expression and provide dense matrix access
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_CAPTURE_H_
#define BCSLIB_MATRIX_CAPTURE_H_

#include <bcslib/matrix/dense_matrix.h>

namespace bcs
{
	template<class Expr, bool ToRef> class matrix_capture;

	template<class Expr>
	class matrix_capture<Expr, true> : private noncopyable
	{
	public:
		typedef Expr captured_type;

		BCS_ENSURE_INLINE
		explicit matrix_capture(const Expr& expr)
		: m_mat(expr) { }

		BCS_ENSURE_INLINE
		const captured_type& get() const
		{
			return m_mat;
		}

	private:
		const Expr& m_mat;
	};


	template<class Expr>
	class matrix_capture<Expr, false> : private noncopyable
	{
	public:
		typedef typename matrix_traits<Expr>::value_type value_type;
		typedef dense_matrix<value_type, ct_rows<Expr>::value, ct_cols<Expr>::value> captured_type;

		BCS_ENSURE_INLINE
		explicit matrix_capture(const Expr& expr)
		: m_mat(expr) { }

		BCS_ENSURE_INLINE
		const captured_type& get() const
		{
			return m_mat;
		}

	private:
		captured_type m_mat;
	};


}

#endif 
