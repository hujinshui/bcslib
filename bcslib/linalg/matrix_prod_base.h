/*
 * @file matrix_prod_base.h
 *
 * Basic facilities for matrix product support
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PROD_BASE_H_
#define BCSLIB_MATRIX_PROD_BASE_H_

#include <bcslib/linalg/matrix_blas.h>
#include <bcslib/matrix/matrix_capture.h>

namespace bcs
{
	template<class LArg, class RArg>
	class matrix_prod_expr_base
	{
	public:
		typedef typename matrix_traits<LArg>::value_type value_type;

		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		matrix_prod_expr_base(const LArg& larg, const RArg& rarg, const value_type& alpha)
		: m_left_arg(larg), m_right_arg(rarg), m_alpha(alpha)
		{
		}

		BCS_ENSURE_INLINE const LArg& left_arg() const { return m_left_arg; }

		BCS_ENSURE_INLINE const RArg& right_arg() const { return m_right_arg; }

		BCS_ENSURE_INLINE value_type alpha() const { return m_alpha; }

	private:
		const LArg& m_left_arg;
		const RArg& m_right_arg;
		const value_type m_alpha;
	};


	struct mm_col_tag { };
	struct mm_row_tag { };
	struct mm_mat_tag { };
	struct mm_tmat_tag { };
	struct mm_smat_tag { };

}

#endif 
