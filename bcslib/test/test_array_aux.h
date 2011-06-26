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


		template<class LIndexer, class RIndexer>
		bool array_view_approx(const aview1d<double, LIndexer>& a1, const aview1d<double, RIndexer>& a2, double eps = 1e-12)
		{
			if (a1.nelems() != a2.nelems()) return false;

			index_t d0 = a1.dim0();
			for (index_t i = 0; i < d0; ++i)
			{
				if (std::abs(a1(i) - a2(i)) > eps) return false;
			}

			return true;
		}


		template<typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
		bool array_view_approx(
				const aview2d<double, TOrd, LIndexer0, LIndexer1>& a1,
				const aview2d<double, TOrd, RIndexer0, RIndexer1>& a2, double eps = 1e-12)
		{
			if (a1.shape() != a2.shape()) return false;

			index_t d0 = a1.dim0();
			index_t d1 = a1.dim1();

			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					if (std::abs(a1(i, j) - a2(i, j)) > eps) return false;
				}
			}

			return true;
		}


		// printing

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


		// accumulation

		// unary

		template<typename T, class TIndexer, class Accumulator>
		typename Accumulator::result_type accum_all(const aview1d<T, TIndexer>& a, Accumulator accum)
		{
			for (index_t i = 0; i < a.dim0(); ++i)
			{
				accum.put(a(i));
			}
			return accum.get();
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1, class Accumulator>
		typename Accumulator::result_type accum_all(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Accumulator accum)
		{
			for (index_t i = 0; i < a.dim0(); ++i)
			{
				for (index_t j = 0; j < a.dim1(); ++j)
				{
					accum.put(a(i, j));
				}
			}
			return accum.get();
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1, class Accumulator>
		array1d<typename Accumulator::result_type> accum_prow(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Accumulator accum0)
		{
			array1d<typename Accumulator::result_type> r(a.nrows());

			for (index_t i = 0; i < a.dim0(); ++i)
			{
				Accumulator accum(accum0);
				for (index_t j = 0; j < a.dim1(); ++j)
				{
					accum.put(a(i, j));
				}
				r(i) = accum.get();
			}
			return r;
		}

		template<typename T, typename TOrd, class TIndexer0, class TIndexer1, class Accumulator>
		array1d<typename Accumulator::result_type> accum_pcol(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Accumulator accum0)
		{
			array1d<typename Accumulator::result_type> r(a.ncolumns());

			for (index_t j = 0; j < a.dim1(); ++j)
			{
				Accumulator accum(accum0);
				for (index_t i = 0; i < a.dim0(); ++i)
				{
					accum.put(a(i, j));
				}
				r(j) = accum.get();
			}
			return r;
		}


		// binary

		template<typename T, class LIndexer, class RIndexer, class Accumulator>
		typename Accumulator::result_type accum_all(
				const aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& b, Accumulator accum)
		{
			for (index_t i = 0; i < a.dim0(); ++i)
			{
				accum.put(a(i), b(i));
			}
			return accum.get();
		}

		template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, class Accumulator>
		typename Accumulator::result_type accum_all(
				const aview2d<T, TOrd, LIndexer0, LIndexer1>& a,
				const aview2d<T, TOrd, RIndexer0, RIndexer1>& b, Accumulator accum)
		{
			for (index_t i = 0; i < a.dim0(); ++i)
			{
				for (index_t j = 0; j < a.dim1(); ++j)
				{
					accum.put(a(i, j), b(i, j));
				}
			}
			return accum.get();
		}

		template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, class Accumulator>
		array1d<typename Accumulator::result_type> accum_prow(
				const aview2d<T, TOrd, LIndexer0, LIndexer1>& a,
				const aview2d<T, TOrd, RIndexer0, RIndexer1>& b, Accumulator accum0)
		{
			array1d<typename Accumulator::result_type> r(a.nrows());

			for (index_t i = 0; i < a.dim0(); ++i)
			{
				Accumulator accum(accum0);
				for (index_t j = 0; j < a.dim1(); ++j)
				{
					accum.put(a(i, j), b(i, j));
				}
				r(i) = accum.get();
			}
			return r;
		}

		template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, class Accumulator>
		array1d<typename Accumulator::result_type> accum_pcol(
				const aview2d<T, TOrd, LIndexer0, LIndexer1>& a,
				const aview2d<T, TOrd, RIndexer0, RIndexer1>& b, Accumulator accum0)
		{
			array1d<typename Accumulator::result_type> r(a.ncolumns());

			for (index_t j = 0; j < a.dim1(); ++j)
			{
				Accumulator accum(accum0);
				for (index_t i = 0; i < a.dim0(); ++i)
				{
					accum.put(a(i, j), b(i, j));
				}
				r(j) = accum.get();
			}
			return r;
		}


	}
}

#endif 
