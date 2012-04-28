/**
 * @file matrix_manip_helpers.h
 *
 * The helpers for implementing matrix manipulation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_MANIP_HELPERS_H_
#define BCSLIB_MATRIX_MANIP_HELPERS_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs { namespace detail {


	/******************************************************
	 *
	 *  equal_helper
	 *
	 ******************************************************/

	template<typename T, int CTLen>
	struct percol_equal_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTLen > 0, "This should only be instantiated with CTLen > 0.");
#endif
		static bool run(const index_t len, const T* __restrict__ a, const T* __restrict__ b)
		{
			return mem<T, CTLen>::equal(a, b);
		}
	};

	template<typename T>
	struct percol_equal_helper<T, DynamicDim>
	{
		static bool run(const index_t len, const T* __restrict__ a, const T* __restrict__ b)
		{
			return elems_equal(len, a, b);
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct matrix_equal_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTCols !=  1, "This should only be instantiated with CTCols != 1.");
#endif

		static const bool IsDynamic = (CTRows == DynamicDim || CTCols == DynamicDim);

		static bool run(const index_t m, const index_t n,
				const T* __restrict__ a, index_t lda,
				const T* __restrict__ b, index_t ldb)
		{
			if (lda == m)
			{
				if (IsDynamic)
				{
					return elems_equal(m * n, a, b);
				}
				else
				{
					return mem<T, CTRows * CTCols>::equal(a, b);
				}
			}
			else
			{
				for (index_t j = 0; j < n; ++j)
				{
					if (!percol_equal_helper<T, CTRows>::run(m, a + j * lda, b + j * ldb))
						return false;
				}
				return true;
			}
		}
	};


	template<typename T, int CTRows>
	struct matrix_equal_helper<T, CTRows, 1>
	{
		static bool run(const index_t m, const index_t n,
				const T* __restrict__ a, index_t lda,
				const T* __restrict__ b, index_t ldb)
		{
			return percol_equal_helper<T, CTRows>::run(m, a, b);
		}
	};


	/******************************************************
	 *
	 *  copy_helper
	 *
	 ******************************************************/

	template<typename T, int CTLen>
	struct percol_copy_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTLen > 0, "This should only be instantiated with CTLen > 0.");
#endif
		static void run(const index_t len, const T* __restrict__ a, T* __restrict__ b)
		{
			mem<T, CTLen>::copy(a, b);
		}
	};

	template<typename T>
	struct percol_copy_helper<T, DynamicDim>
	{
		static void run(const index_t len, const T* __restrict__ a, T* __restrict__ b)
		{
			copy_elems(len, a, b);
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct matrix_copy_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTCols !=  1, "This should only be instantiated with CTCols != 1.");
#endif

		static const bool IsDynamic = (CTRows == DynamicDim || CTCols == DynamicDim);

		static void run(const index_t m, const index_t n,
				const T* __restrict__ a, index_t lda,
				      T* __restrict__ b, index_t ldb)
		{
			if (lda == m)
			{
				if (IsDynamic)
				{
					copy_elems(m * n, a, b);
				}
				else
				{
					mem<T, CTRows * CTCols>::copy(a, b);
				}
			}
			else
			{
				for (index_t j = 0; j < n; ++j)
				{
					percol_copy_helper<T, CTRows>::run(m, a + j * lda, b + j * ldb);
				}
			}
		}
	};


	template<typename T, int CTRows>
	struct matrix_copy_helper<T, CTRows, 1>
	{
		static void run(const index_t m, const index_t n,
				const T* __restrict__ a, index_t lda,
				      T* __restrict__ b, index_t ldb)
		{
			percol_copy_helper<T, CTRows>::run(m, a, b);
		}
	};




	/******************************************************
	 *
	 *  fill_helper
	 *
	 ******************************************************/

	template<typename T, int CTLen>
	struct percol_fill_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTLen > 0, "This should only be instantiated with CTLen > 0.");
#endif
		static void run(const index_t len, T* __restrict__ a, const T& v)
		{
			mem<T, CTLen>::fill(a, v);
		}
	};

	template<typename T>
	struct percol_fill_helper<T, DynamicDim>
	{
		static void run(const index_t len, T* __restrict__ a, const T& v)
		{
			fill_elems(len, a, v);
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct matrix_fill_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTCols !=  1, "This should only be instantiated with CTCols != 1.");
#endif

		static const bool IsDynamic = (CTRows == DynamicDim || CTCols == DynamicDim);

		static void run(const index_t m, const index_t n, T* __restrict__ a, index_t lda, const T& v)
		{
			if (lda == m)
			{
				if (IsDynamic)
				{
					fill_elems(m * n, a, v);
				}
				else
				{
					mem<T, CTRows * CTCols>::fill(a, v);
				}
			}
			else
			{
				for (index_t j = 0; j < n; ++j)
				{
					percol_fill_helper<T, CTRows>::run(m, a + j * lda, v);
				}
			}
		}
	};


	template<typename T, int CTRows>
	struct matrix_fill_helper<T, CTRows, 1>
	{
		static void run(const index_t m, const index_t n, T* __restrict__ a, index_t lda, const T& v)
		{
			percol_fill_helper<T, CTRows>::run(m, a, v);
		}
	};


	/******************************************************
	 *
	 *  zero_helper
	 *
	 ******************************************************/

	template<typename T, int CTLen>
	struct percol_zero_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTLen > 0, "This should only be instantiated with CTLen > 0.");
#endif
		static void run(const index_t len, T* __restrict__ a)
		{
			mem<T, CTLen>::zero(a);
		}
	};

	template<typename T>
	struct percol_zero_helper<T, DynamicDim>
	{
		static void run(const index_t len, T* __restrict__ a)
		{
			zero_elems(len, a);
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct matrix_zero_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTCols !=  1, "This should only be instantiated with CTCols != 1.");
#endif

		static const bool IsDynamic = (CTRows == DynamicDim || CTCols == DynamicDim);

		static void run(const index_t m, const index_t n, T* __restrict__ a, index_t lda)
		{
			if (lda == m)
			{
				if (IsDynamic)
				{
					zero_elems(m * n, a);
				}
				else
				{
					mem<T, CTRows * CTCols>::zero(a);
				}
			}
			else
			{
				for (index_t j = 0; j < n; ++j)
				{
					percol_zero_helper<T, CTRows>::run(m, a + j * lda);
				}
			}
		}
	};


	template<typename T, int CTRows>
	struct matrix_zero_helper<T, CTRows, 1>
	{
		static void run(const index_t m, const index_t n, T* __restrict__ a, index_t lda)
		{
			percol_zero_helper<T, CTRows>::run(m, a);
		}
	};


} }



#endif /* MATRIX_MANIP_HELPERS_H_ */
