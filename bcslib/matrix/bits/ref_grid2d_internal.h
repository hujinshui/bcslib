/**
 * @file ref_matrix_ex_internal.h
 *
 * The internal implementation of ref_grid2d
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_GRID2D_INTERNAL_H_
#define BCSLIB_REF_GRID2D_INTERNAL_H_

#include <bcslib/matrix/matrix_properties.h>
#include <bcslib/matrix/bits/offset_helper.h>


namespace bcs { namespace detail {

	template<bool IsRow, bool IsCol> struct grid2d_offset_helper;

	template<> struct grid2d_offset_helper<false, false>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t lin(const Mat& mat, const index_t)
		{
			throw invalid_operation("Attempted to access a grid2d (that is not a compile-time vector) with linear index.");
		}

		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t sub(const Mat& mat, const index_t i, const index_t j)
		{
			return i * mat.inner_step() + j * mat.lead_dim();
		}
	};

	template<> struct grid2d_offset_helper<false, true>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t lin(const Mat& mat, const index_t i)
		{
			return i * mat.inner_step();
		}

		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t sub(const Mat& mat, const index_t i, const index_t j)
		{
			return i * mat.inner_step();
		}
	};


	template<> struct grid2d_offset_helper<true, false>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t lin(const Mat& mat, const index_t i)
		{
			return i * mat.lead_dim();
		}

		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t sub(const Mat& mat, const index_t i, const index_t j)
		{
			return j * mat.lead_dim();
		}
	};


	template<> struct grid2d_offset_helper<true, true>
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t lin(const Mat& mat, const index_t)
		{
			return 0;
		}

		template<class Mat>
		BCS_ENSURE_INLINE
		static index_t sub(const Mat& mat, const index_t i, const index_t j)
		{
			return 0;
		}
	};



	template<typename T, int CTRows, int CTCols>
	class ref_grid2d_internal
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTRows >= 1 && CTCols >= 1,
				"The values of CTRows and CTCols must be positive.");
#endif

		typedef grid2d_offset_helper<CTRows == 1, CTCols == 1> offset_helper_t;

	public:
		BCS_ENSURE_INLINE
		ref_grid2d_internal(T *pdata, index_t m, index_t n, index_t step, index_t ldim)
		: m_pdata(pdata), m_inner_step(step), m_leaddim(ldim)
		{
			check_arg(m == CTRows && n == CTCols,
					"Attempted to construct a ref_matrix_ex with incorrect dimensions.");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t inner_step() const { return m_inner_step; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

		BCS_ENSURE_INLINE index_t lin_offset(const index_t i) const
		{
			return offset_helper_t::lin(*this, i);
		}

		BCS_ENSURE_INLINE index_t sub_offset(const index_t i, const index_t j) const
		{
			return offset_helper_t::sub(*this, i, j);
		}

	private:
		T* m_pdata;
		index_t m_inner_step;
		index_t m_leaddim;
	};

	template<typename T, int CTCols>
	class ref_grid2d_internal<T, DynamicDim, CTCols>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTCols >= 1, "The values of CTCols must be positive.");
#endif

		typedef grid2d_offset_helper<false, CTCols == 1> offset_helper_t;

	public:
		BCS_ENSURE_INLINE
		ref_grid2d_internal(T *pdata, index_t m, index_t n, index_t step, index_t ldim)
		: m_pdata(pdata), m_nrows(m), m_inner_step(step), m_leaddim(ldim)
		{
			check_arg(n == CTCols,
					"Attempted to construct a ref_matrix_ex with incorrect dimension (n != CTCols)");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t inner_step() const { return m_inner_step; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

		BCS_ENSURE_INLINE index_t lin_offset(const index_t i) const
		{
			return offset_helper_t::lin(*this, i);
		}

		BCS_ENSURE_INLINE index_t sub_offset(const index_t i, const index_t j) const
		{
			return offset_helper_t::sub(*this, i, j);
		}

	private:
		T* m_pdata;
		index_t m_nrows;
		index_t m_inner_step;
		index_t m_leaddim;
	};


	template<typename T, int CTRows>
	class ref_grid2d_internal<T, CTRows, DynamicDim>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTRows >= 1, "The values of CTCols must be positive.");
#endif

		typedef grid2d_offset_helper<CTRows == 1, false> offset_helper_t;

	public:
		BCS_ENSURE_INLINE
		ref_grid2d_internal(T *pdata, index_t m, index_t n, index_t step, index_t ldim)
		: m_pdata(pdata), m_ncols(n), m_inner_step(step), m_leaddim(ldim)
		{
			check_arg(m == CTRows,
					"Attempted to construct a ref_matrix_ex with incorrect dimension (m != CTRows)");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t inner_step() const { return m_inner_step; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

		BCS_ENSURE_INLINE index_t lin_offset(const index_t i) const
		{
			return offset_helper_t::lin(*this, i);
		}

		BCS_ENSURE_INLINE index_t sub_offset(const index_t i, const index_t j) const
		{
			return offset_helper_t::sub(*this, i, j);
		}

	private:
		T* m_pdata;
		index_t m_ncols;
		index_t m_inner_step;
		index_t m_leaddim;
	};


	template<typename T>
	class ref_grid2d_internal<T, DynamicDim, DynamicDim>
	{
		typedef grid2d_offset_helper<false, false> offset_helper_t;

	public:
		BCS_ENSURE_INLINE
		ref_grid2d_internal(T *pdata, index_t m, index_t n, index_t step, index_t ldim)
		: m_pdata(pdata), m_nrows(m), m_ncols(n)
		, m_inner_step(step), m_leaddim(ldim)
		{
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t inner_step() const { return m_inner_step; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

		BCS_ENSURE_INLINE index_t lin_offset(const index_t i) const
		{
			return offset_helper_t::lin(*this, i);
		}

		BCS_ENSURE_INLINE index_t sub_offset(const index_t i, const index_t j) const
		{
			return offset_helper_t::sub(*this, i, j);
		}

	private:
		T* m_pdata;
		index_t m_nrows;
		index_t m_ncols;
		index_t m_inner_step;
		index_t m_leaddim;
	};


} }

#endif /* REF_MATRIX_INTERNAL_H_ */




