/**
 * @file test_assertion.h
 *
 * A collection of assertion and predicates for testing
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_TEST_ASSERTION_H
#define BCSLIB_TEST_ASSERTION_H

#include <bcslib/base/basic_defs.h>
// #include <bcslib/array/array1d.h>
// #include <bcslib/array/array2d.h>

#include <cstdio>
#include <string>

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


		// array comparison

/*

*/

	}
}



// useful assertion macros

#define BCS_CHECK( condexpr ) \
	if (!(condexpr)) { throw bcs::test::assertion_failure(__FILE__, __LINE__, #condexpr); }

#define BCS_CHECK_MESSAGE( condexpr, msg ) \
	if (!(condexpr)) { throw bcs::test::assertion_failure(__FILE__, __LINE__, msg); }

#define BCS_CHECK_EQUAL( lhs, rhs ) BCS_CHECK_MESSAGE( (lhs) == (rhs), #lhs " == " #rhs )

#define BCS_CHECK_APPROX_( lhs, rhs, eps ) BCS_CHECK_MESSAGE( test_approx(lhs, rhs, eps), #lhs " ~= " #rhs)
#define BCS_CHECK_APPROX( lhs, rhs ) BCS_CHECK_APPROX_( lhs, rhs, 1e-12 )


#endif 
