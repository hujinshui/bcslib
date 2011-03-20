/**
 * @file test_assertion.h
 *
 * A collection of assertion and predicates for testing
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TEST_ASSERTION_H
#define BCSLIB_TEST_ASSERTION_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>
#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <cstdio>
#include <string>
#include <algorithm>

namespace bcs
{
	namespace test
	{
		class assertion_failure
		{
		public:
			assertion_failure(const char *fname, int ln, const char *msg)
			: m_filename(fname), m_lineno(ln), m_message(msg)
			{
			}

			std::string filename() const
			{
				return m_filename;
			}

			int line_number() const
			{
				return m_lineno;
			}

			std::string message() const
			{
				return m_message;
			}

		private:
			std::string m_filename;
			int m_lineno;
			std::string m_message;

		}; // end class assertion_failure


		// useful predicates

		template<typename T>
		inline bool test_approx(const T& x1, const T& x2, const T& eps)
		{
			return x1 >= x2 ? x1 < x2 + eps : x1 > x2 - eps;
		}


		template<typename TIter0, typename TIter>
		bool collection_equal(TIter0 begin, TIter0 end, TIter src, size_t n)
		{
			TIter0 it = begin;
			for (size_t i = 0; i < n; ++i)
			{
				if (it == end)
				{
					return false;
				}
				if (*it != *src)
				{
					return false;
				}

				++it;
				++src;
			}

			return it == end;
		}


		template<typename TEnum, typename TIter>
		bool enumerate_equal(TEnum e, TIter it, size_t n)
		{
			size_t c = 0;

			while (c < n && e.next())
			{
				if (e.get() != *it++) return false;
				++ c;
			}

			if (c < n) return false;
			if (e.next()) return false;

			return true;
		}


		// array comparison


		template<typename T, class TIndexer>
		bool array_view_equal(const bcs::const_aview1d<T, TIndexer>& view, const T* src, size_t n)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (view(i) != src[i]) return false;
			}

			return true;
		}

		template<typename T, class TIndexer>
		bool array_view_approx(const bcs::const_aview1d<T, TIndexer>& view, const T* src, size_t n, double eps = 1e-12)
		{
			if (view.nelems() != n) return false;

			for (index_t i = 0; i < (index_t)n; ++i)
			{
				if (!test_approx(view(i), src[i], eps)) return false;
			}

			return true;
		}


		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_equal(const bcs::const_aview2d<T, row_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.dim0() == m && view.dim1() == n)) return false;

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
		bool array_view_equal(const bcs::const_aview2d<T, column_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n)
		{
			if (!(view.dim0() == m && view.dim1() == n)) return false;

			for (index_t j = 0; j < (index_t)n; ++j)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					if (view(i, j) != *src++) return false;
				}
			}

			return true;
		}

		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_approx(const bcs::const_aview2d<T, row_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n, double eps = 1e-12)
		{
			if (!(view.dim0() == m && view.dim1() == n)) return false;

			for (index_t i = 0; i < (index_t)m; ++i)
			{
				for (index_t j = 0; j < (index_t)n; ++j)
				{
					if (!test_approx(view(i, j), *src++)) return false;
				}
			}

			return true;
		}

		template<typename T, class TIndexer0, class TIndexer1>
		bool array_view_approx(const bcs::const_aview2d<T, column_major_t, TIndexer0, TIndexer1>& view, const T* src, size_t m, size_t n, double eps=1e-12)
		{
			if (!(view.dim0() == m && view.dim1() == n)) return false;

			for (index_t j = 0; j < (index_t)n; ++j)
			{
				for (index_t i = 0; i < (index_t)m; ++i)
				{
					if (!test_approx(view(i, j), *src++)) return false;
				}
			}

			return true;
		}


	}
}



// useful assertion macros

#define BCS_CHECK( condexpr ) \
	if (!(condexpr)) { throw bcs::test::assertion_failure(__FILE__, __LINE__, #condexpr); }

#define BCS_CHECK_MESSAGE( condexpr, msg ) \
	if (!(condexpr)) { throw bcs::test::assertion_failure(__FILE__, __LINE__, msg); }

#define BCS_CHECK_EQUAL( lhs, rhs ) BCS_CHECK_MESSAGE( (lhs) == (rhs), #lhs " == " #rhs )

#define BCS_CHECK_APPROX( lhs, rhs, eps ) BCS_CHECK_MESSAGE( test_approx, #lhs " ~= " #rhs)


#endif 
