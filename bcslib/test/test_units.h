/**
 * @file test_units.h
 *
 * The class to represent test units and test suites, which are
 * the basic building block of our unit testing framework.
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TEST_UNITS_H
#define BCSLIB_TEST_UNITS_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/test/test_assertion.h>

#include <vector>
#include <string>

namespace bcs
{
	namespace test
	{
		class base_test_unit
		{
		public:
			base_test_unit(const char *name) : m_name(name) { }

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

			void add(base_test_unit *tunit)
			{
				m_tunits.push_back(tr1::shared_ptr<base_test_unit>(tunit));
				m_ncases += tunit->num_cases();
			}

			virtual int num_cases() const
			{
				return m_ncases;
			}

			base_test_unit* get_unit(int i) const
			{
				return m_tunits[i].get();
			}

		private:
			std::vector<tr1::shared_ptr<base_test_unit> > m_tunits;
			int m_ncases;

		}; // end class test_suite

	}
}


// Macros to faciliate the definition of test case

#define BCS_TEST_CASE( case_name ) \
	class case_name : public base_test_case { \
		public: \
			case_name() : base_test_case( #case_name ) { } \
			virtual void run(); \
	}; \
	void case_name::run()

#endif 
