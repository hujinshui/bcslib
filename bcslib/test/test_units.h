/**
 * @file test_units.h
 *
 * The class to represent test units and test suites, which are
 * the basic building block of our unit testing framework.
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_TEST_UNITS_H
#define BCSLIB_TEST_UNITS_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/test/test_assertion.h>

#include <vector>
#include <string>
#include <memory>

namespace bcs
{
	namespace test
	{
		class base_test_unit
		{
		public:
			base_test_unit(const char *name) : m_name(name) { }

			virtual ~base_test_unit() { }

			std::string name() const
			{
				return m_name;
			}

			virtual int num_cases() const = 0;

		private:
			std::string m_name;

		}; // end class test_unit


		class base_test_case : public base_test_unit
		{
		public:
			base_test_case(const char *name) : base_test_unit(name) { }

			virtual ~base_test_case() { }

			virtual void run() = 0;

			virtual int num_cases() const { return 1; }

		}; // end class base_test_case


		class test_suite : public base_test_unit
		{
		public:
			test_suite(const char *name)
			: base_test_unit(name), m_ncases(0)
			{
			}

			virtual ~test_suite()
			{
			}

			void add(std::shared_ptr<base_test_unit> tunit)
			{
				m_tunits.push_back(tunit);
				m_ncases += tunit->num_cases();
			}

			virtual int num_cases() const
			{
				return m_ncases;
			}

			int num_units() const
			{
				return (int)m_tunits.size();
			}

			base_test_unit* get_unit(int i) const
			{
				return m_tunits[(size_t)i].get();
			}

		private:
			std::vector<std::shared_ptr<base_test_unit> > m_tunits;
			int m_ncases;

		}; // end class test_suite

	}
}


// Macros to faciliate the definition of test case

#define BCS_TEST_CASE( case_name ) \
	class case_name : public base_test_case { \
		public: \
			case_name() : base_test_case( #case_name ) { } \
			virtual ~case_name() { } \
			virtual void run(); \
	}; \
	void case_name::run()

#define BCS_NEW_TEST_SUITE( suite, title ) std::shared_ptr<bcs::test::test_suite> suite( new bcs::test::test_suite(title) )

#define BCS_ADD_TEST_CASE( suite, case_cons ) suite->add( std::shared_ptr<bcs::test::base_test_case>(new case_cons) )


#endif 
