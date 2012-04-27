/**
 * @file matrix_helpers.h
 *
 * Some helpful devices for matrix implementation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_HELPERS_H_
#define BCSLIB_MATRIX_HELPERS_H_

#include <bcslib/core/basic_defs.h>

namespace bcs
{
	namespace detail
	{
		template<bool SingleRow, bool SingleCol> struct dim_helper;

		template<> struct dim_helper<false, false>
		{
			BCS_ENSURE_INLINE static index_t nelems(index_t m, index_t n)
			{
				return m * n;
			}

			BCS_ENSURE_INLINE static bool is_empty(index_t m, index_t n)
			{
				return m == 0 || n == 0;
			}

			BCS_ENSURE_INLINE static bool is_scalar(index_t m, index_t n)
			{
				return m == 1 && n == 1;
			}

			BCS_ENSURE_INLINE static bool is_vector(index_t m, index_t n)
			{
				return m == 1 || n == 1;
			}

			BCS_ENSURE_INLINE static bool is_row(index_t m)
			{
				return m == 1;
			}

			BCS_ENSURE_INLINE static bool is_column(index_t n)
			{
				return n == 1;
			}

			BCS_ENSURE_INLINE static index_t offset(const index_t leadim, index_t i, index_t j)
			{
				return i + leadim * j;
			}
		};


		template<> struct dim_helper<false, true>
		{
			BCS_ENSURE_INLINE static index_t nelems(index_t m, index_t n)
			{
				return m;
			}

			BCS_ENSURE_INLINE static bool is_empty(index_t m, index_t n)
			{
				return m == 0;
			}

			BCS_ENSURE_INLINE static bool is_scalar(index_t m, index_t n)
			{
				return m == 1;
			}

			BCS_ENSURE_INLINE static bool is_vector(index_t m, index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_row(index_t m)
			{
				return m == 1;
			}

			BCS_ENSURE_INLINE static bool is_column(index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static index_t offset(const index_t leaddim, index_t i, index_t j)
			{
				return i;
			}
		};


		template<> struct dim_helper<true, false>
		{
			BCS_ENSURE_INLINE static index_t nelems(index_t m, index_t n)
			{
				return n;
			}

			BCS_ENSURE_INLINE static bool is_empty(index_t m, index_t n)
			{
				return n == 0;
			}

			BCS_ENSURE_INLINE static bool is_scalar(index_t m, index_t n)
			{
				return n == 1;
			}

			BCS_ENSURE_INLINE static bool is_vector(index_t m, index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_row(index_t m)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_column(index_t n)
			{
				return n == 1;
			}

			BCS_ENSURE_INLINE static index_t offset(const index_t leaddim, index_t i, index_t j)
			{
				return leaddim * j;
			}
		};


		template<> struct dim_helper<true, true>
		{
			BCS_ENSURE_INLINE static index_t nelems(index_t m, index_t n)
			{
				return 1;
			}

			BCS_ENSURE_INLINE static bool is_empty(index_t m, index_t n)
			{
				return false;
			}

			BCS_ENSURE_INLINE static bool is_scalar(index_t m, index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_vector(index_t m, index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_row(index_t m)
			{
				return true;
			}

			BCS_ENSURE_INLINE static bool is_column(index_t n)
			{
				return true;
			}

			BCS_ENSURE_INLINE static index_t offset(const index_t leaddim, index_t i, index_t j)
			{
				return 0;
			}
		};


	}

}

#endif /* MATRIX_HELPERS_H_ */
