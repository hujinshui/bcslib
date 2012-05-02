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

#ifndef BCSLIB_MATRIX_MEMOP_INTERNAL_H_
#define BCSLIB_MATRIX_MEMOP_INTERNAL_H_

#include <bcslib/core/mem_op.h>
#include <bcslib/matrix/matrix_concepts.h>

namespace bcs { namespace detail {


	/******************************************************
	 *
	 *  equal_helper
	 *
	 ******************************************************/

	template<typename T, class LMat, class RMat>
	struct equal_helper_directmem
	{
		static const int Size = binary_ct_size<LMat, RMat>::value;

		BCS_ENSURE_INLINE
		static bool test(const LMat& a, const RMat& b)
		{
			if (Size > 0)
			{
				return mem<T, Size>::equal(a.ptr_data(), b.ptr_data());
			}
			else
			{
				return elems_equal(a.nelems(), a.ptr_data(), b.ptr_data());
			}
		}
	};


	template<typename T, class LMat, class RMat>
	struct equal_helper_colwise_directmem
	{
		static const int CTRows = ct_rows<LMat>::value;

		inline static bool test(const LMat& a, const RMat& b)
		{
			const index_t n = a.ncolumns();
			if (CTRows > 0)
			{
				if (n == 1)
				{
					return mem<T, CTRows>::equal(a.ptr_data(), b.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						if (!mem<T, CTRows>::equal(col_ptr(a, j), col_ptr(b, j)))
							return false;
					}
					return true;
				}
			}
			else
			{
				const index_t m = a.nrows();

				if (n == 1)
				{
					return elems_equal(m, a.ptr_data(), b.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						if (!elems_equal(m, col_ptr(a, j), col_ptr(b, j)))
							return false;
					}
					return true;
				}
			}
		}
	};


	template<typename T, class LMat, class RMat>
	struct equal_helper_regular
	{
		inline static bool test(const LMat& a, const RMat& b)
		{
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					if (a(i, j) != b(i, j)) return false;
				}
			}
			return true;
		}
	};

	template<class LMat, class RMat>
	struct equal_helper
	{
		typedef typename matrix_traits<LMat>::value_type T;

		static const bool AreBothDense = is_dense_mat<LMat>::value && is_dense_mat<RMat>::value;
		static const bool AreBothCont = is_continuous_mat<LMat>::value && is_continuous_mat<RMat>::value;

		typedef typename select_type<AreBothDense,
					typename select_type<AreBothCont,
						equal_helper_directmem<T, LMat, RMat>,
						equal_helper_colwise_directmem<T, LMat, RMat> >::type,
					equal_helper_regular<T, LMat, RMat> >::type type;
	};


	/******************************************************
	 *
	 *  copy_helper
	 *
	 ******************************************************/

	template<typename T, class LMat, class RMat>
	struct copy_helper_directmem
	{
		static const int Size = binary_ct_size<LMat, RMat>::value;

		BCS_ENSURE_INLINE
		static void run(const LMat& a, RMat& b)
		{
			if (Size > 0)
			{
				mem<T, Size>::copy(a.ptr_data(), b.ptr_data());
			}
			else
			{
				copy_elems(a.nelems(), a.ptr_data(), b.ptr_data());
			}
		}
	};

	template<typename T, class LMat, class RMat>
	struct copy_helper_colwise_directmem
	{
		static const int CTRows = ct_rows<LMat>::value;

		inline static void run(const LMat& a, RMat& b)
		{
			const index_t n = a.ncolumns();
			if (CTRows > 0)
			{
				if (n == 1)
				{
					mem<T, CTRows>::copy(a.ptr_data(), b.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						mem<T, CTRows>::copy(col_ptr(a, j), col_ptr(b, j));
					}
				}
			}
			else
			{
				const index_t m = a.nrows();

				if (n == 1)
				{
					copy_elems(m, a.ptr_data(), b.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						copy_elems(m, col_ptr(a, j), col_ptr(b, j));
					}
				}
			}
		}
	};


	template<typename T, class LMat, class RMat>
	struct copy_helper_regular
	{
		inline static void run(const LMat& a, RMat& b)
		{
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					b(i, j) = a(i, j);
				}
			}
		}
	};


	template<class LMat, class RMat>
	struct copy_helper
	{
		typedef typename matrix_traits<LMat>::value_type T;

		static const bool AreBothDense = is_dense_mat<LMat>::value && is_dense_mat<RMat>::value;
		static const bool AreBothCont = is_continuous_mat<LMat>::value && is_continuous_mat<RMat>::value;

		typedef typename select_type<AreBothDense,
					typename select_type<AreBothCont,
						copy_helper_directmem<T, LMat, RMat>,
						copy_helper_colwise_directmem<T, LMat, RMat> >::type,
					copy_helper_regular<T, LMat, RMat> >::type type;
	};



	/******************************************************
	 *
	 *  fill_helper
	 *
	 ******************************************************/

	template<typename T, class DMat>
	struct fill_helper_directmem
	{
		static const int Size = ct_size<DMat>::value;

		BCS_ENSURE_INLINE
		static void run(DMat& a, const T v)
		{
			if (Size > 0)
			{
				mem<T, Size>::fill(a.ptr_data(), v);
			}
			else
			{
				fill_elems(a.nelems(), a.ptr_data(), v);
			}
		}
	};

	template<typename T, class DMat>
	struct fill_helper_colwise_directmem
	{
		static const int CTRows = ct_rows<DMat>::value;

		inline static void run(DMat& a, const T v)
		{
			const index_t n = a.ncolumns();
			if (CTRows > 0)
			{
				if (n == 1)
				{
					mem<T, CTRows>::fill(a.ptr_data(), v);
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						mem<T, CTRows>::fill(col_ptr(a, j), v);
					}
				}
			}
			else
			{
				const index_t m = a.nrows();

				if (n == 1)
				{
					fill_elems(m, a.ptr_data(), v);
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						fill_elems(m, col_ptr(a, j), v);
					}
				}
			}
		}
	};


	template<typename T, class DMat>
	struct fill_helper_regular
	{
		inline static void run(DMat& a, const T v)
		{
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					a(i, j) = v;
				}
			}
		}
	};

	template<class DMat>
	struct fill_helper
	{
		typedef typename matrix_traits<DMat>::value_type T;

		typedef typename select_type<is_dense_mat<DMat>::value,
					typename select_type<is_continuous_mat<DMat>::value,
						fill_helper_directmem<T, DMat>,
						fill_helper_colwise_directmem<T, DMat> >::type,
					fill_helper_regular<T, DMat> >::type type;
	};



	/******************************************************
	 *
	 *  zero_helper
	 *
	 ******************************************************/

	template<typename T, class DMat>
	struct zero_helper_directmem
	{
		static const int Size = ct_size<DMat>::value;

		BCS_ENSURE_INLINE
		static void run(DMat& a)
		{
			if (Size > 0)
			{
				mem<T, Size>::zero(a.ptr_data());
			}
			else
			{
				zero_elems(a.nelems(), a.ptr_data());
			}
		}
	};

	template<typename T, class DMat>
	struct zero_helper_colwise_directmem
	{
		static const int CTRows = ct_rows<DMat>::value;

		inline static void run(DMat& a)
		{
			const index_t n = a.ncolumns();
			if (CTRows > 0)
			{
				if (n == 1)
				{
					mem<T, CTRows>::zero(a.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						mem<T, CTRows>::zero(col_ptr(a, j));
					}
				}
			}
			else
			{
				const index_t m = a.nrows();

				if (n == 1)
				{
					zero_elems(m, a.ptr_data());
				}
				else
				{
					for (index_t j = 0; j < n; ++j)
					{
						zero_elems(m, col_ptr(a, j));
					}
				}
			}
		}
	};


	template<typename T, class DMat>
	struct zero_helper_regular
	{
		inline static void run(DMat& a)
		{
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					a(i, j) = T(0);
				}
			}
		}
	};

	template<class DMat>
	struct zero_helper
	{
		typedef typename matrix_traits<DMat>::value_type T;

		typedef typename select_type<is_dense_mat<DMat>::value,
					typename select_type<is_continuous_mat<DMat>::value,
						zero_helper_directmem<T, DMat>,
						zero_helper_colwise_directmem<T, DMat> >::type,
					zero_helper_regular<T, DMat> >::type type;
	};


} }



#endif /* MATRIX_MANIP_HELPERS_H_ */
