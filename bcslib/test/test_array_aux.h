/**
 * @file test_array_aux.h
 *
 * Auxiliary facilities for array-based testing
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TEST_ARRAY_AUX_H_
#define BCSLIB_TEST_ARRAY_AUX_H_

#include <bcslib/test/test_assertion.h>
#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <cmath>
#include <cstdio>

namespace bcs
{
	namespace test
	{
		template<typename T>
		bool array_view_equal(const caview1d<T>& view, const T* src, size_t n)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (view(i) != src[i]) return false;
			}

			return true;
		}

		template<typename T, class TIndexer>
		bool array_view_equal(const caview1d_ex<T, TIndexer>& view, const T* src, size_t n)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (view(i) != src[i]) return false;
			}

			return true;
		}


		template<typename T, typename TOrd>
		bool array_view_equal(const caview2d<T, TOrd>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			if (std::is_same<TOrd, row_major_t>::value)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					for (index_t j = 0; j < (index_t)n; ++j)
					{
						if (view(i, j) != *src++) return false;
					}
				}
			}
			else
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					for (index_t i = 0; i < (index_t)m; ++i)
					{
						if (view(i, j) != *src++) return false;
					}
				}
			}

			return true;
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		bool array_view_equal(const caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			if (std::is_same<TOrd, row_major_t>::value)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					for (index_t j = 0; j < (index_t)n; ++j)
					{
						if (view(i, j) != *src++) return false;
					}
				}
			}
			else
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					for (index_t i = 0; i < (index_t)m; ++i)
					{
						if (view(i, j) != *src++) return false;
					}
				}
			}

			return true;
		}

		template<typename T, typename TOrd>
		bool array_view_equal(const caview2d<T, TOrd>& view, const T& v, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t i = 0; i < (index_t)m; ++i)
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					if (view(i, j) != v) return false;
				}
			}

			return true;
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		bool array_view_equal(const caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& view, const T& v, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t i = 0; i < (index_t)m; ++i)
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					if (view(i, j) != v) return false;
				}
			}

			return true;
		}


		// printing

		template<typename T>
		void print_array(const bcs::caview1d<T>& view, const char *fmt = "%g ")
		{
			index_t n = (index_t)view.nelems();
			for (index_t i = 0; i < n; ++i)
			{
				std::printf(fmt, view(i));
			}
			std::printf("\n");
		}

		template<typename T, class TIndexer>
		void print_array(const bcs::caview1d_ex<T, TIndexer>& view, const char *fmt = "%g ")
		{
			index_t n = (index_t)view.nelems();
			for (index_t i = 0; i < n; ++i)
			{
				std::printf(fmt, view(i));
			}
			std::printf("\n");
		}


		template<typename T, typename TOrd>
		void print_array(const bcs::caview2d<T, TOrd>& view, const char *fmt = "%g ")
		{
			index_t m = (index_t)view.nrows();
			index_t n = (index_t)view.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					std::printf(fmt, view(i, j));
				}
				std::printf("\n");
			}
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		void print_array(const bcs::caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& view, const char *fmt = "%g ")
		{
			index_t m = (index_t)view.nrows();
			index_t n = (index_t)view.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					std::printf(fmt, view(i, j));
				}
				std::printf("\n");
			}
		}

	}
}

#endif 
