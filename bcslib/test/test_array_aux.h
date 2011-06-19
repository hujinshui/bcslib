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

#include <iostream>

namespace bcs
{
	namespace test
	{
		template<typename T, class TIndexer>
		bool array_view_equal(const aview1d<T, TIndexer>& view, const T* src, size_t n)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (view(i) != src[i]) return false;
			}

			return true;
		}

		template<typename T, class TIndexer>
		bool array_view_approx(const aview1d<T, TIndexer>& view, const T* src, size_t n, double eps = 1e-12)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (!test_approx(view(i), src[i], eps)) return false;
			}

			return true;
		}


		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_equal(const aview2d<T, row_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t i = 0; i < (index_t)m; ++i)
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					if (view(i, j) != *src++) return false;
				}
			}

			return true;
		}

		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_equal(const aview2d<T, column_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t j = 0; j < (index_t)n; ++j)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					if (view(i, j) != *src++) return false;
				}
			}

			return true;
		}


		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		bool array_view_equal(const aview2d<T, TOrd, TIndexer0, TIndexer1>& view, const T& v, size_t m, size_t n)
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


		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_approx(const bcs::aview2d<T, row_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n, double eps = 1e-12)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t i = 0; i < (index_t)m; ++i)
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					if (!test_approx(view(i, j), *src++, eps)) return false;
				}
			}

			return true;
		}

		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_approx(const bcs::aview2d<T, column_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n, double eps=1e-12)
		{
			if (!(view.nrows() == m && view.ncolumns() == n)) return false;

			for (index_t j = 0; j < (index_t)n; ++j)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					if (!test_approx(view(i, j), *src++, eps)) return false;
				}
			}

			return true;
		}


		template<typename T, class TIndexer>
		void print_array(const bcs::aview1d<T, TIndexer>& view, const char *title = 0)
		{
			if (title != 0)
				std::cout << title << ' ';

			index_t n = (index_t)view.nelems();
			for (index_t i = 0; i < n; ++i)
			{
				std::cout << view(i) << ' ';
			}
			std::cout << std::endl;
		}


		template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
		void print_array(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
		{
			index_t m = (index_t)view.nrows();
			index_t n = (index_t)view.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					std::cout << view(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}

	}
}

#endif 
