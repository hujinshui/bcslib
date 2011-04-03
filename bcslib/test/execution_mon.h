/**
 * @file execution_mon.h
 *
 * The test execution monitors
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_EXECUTION_MON_H
#define BCSLIB_EXECUTION_MON_H

#include <bcslib/test/test_assertion.h>
#include <bcslib/test/test_units.h>

#include <cstdio>

namespace bcs
{
	namespace test
	{

		class base_execution_monitor
		{
		public:
			virtual ~base_execution_monitor()
			{
			}

			virtual void execute(test_suite* tsuite) = 0;

		}; // end class base_execution_monitor


		class stdc_execution_monitor
		{
		public:
			virtual ~stdc_execution_monitor()
			{
			}

			virtual void execute(test_suite* tsuite)
			{
				std::printf("BCS Testing\n");
				std::printf("----------------------------------------\n");

				int nsucc = execute_suite("MAIN", 1, 1, tsuite);
				int nfail = tsuite->num_cases() - nsucc;

				if (nfail == 0)
				{
					std::printf("All %d cases pass test!\n", tsuite->num_cases());
				}
				else
				{
					std::printf("In total, %d out of %d cases failed.\n", nfail, tsuite->num_cases());
				}
				std::printf("\n");
			}

		private:
			int execute_suite(const std::string& parent, int i0, int n0, test_suite* tsuite)
			{
				int n = tsuite->num_units();
				std::printf("[%s %d/%d] test suite: %s (with %d cases) ...\n",
						parent.c_str(), i0, n0, tsuite->name().c_str(), tsuite->num_cases());

				int nsucc = 0;

				for (int i = 0; i < n; ++i)
				{
					base_test_unit *tunit = tsuite->get_unit(i);

					test_suite *ts = dynamic_cast<test_suite*>(tunit);
					if (ts != 0)
					{
						nsucc += execute_suite(tsuite->name(), i+1, n, ts);
					}
					else
					{
						base_test_case *tc = dynamic_cast<base_test_case*>(tunit);
						if (tc != 0)
						{
							if (execute_case(tsuite->name(), i+1, n, tc)) ++ nsucc;
						}
					}
				}

				return nsucc;
			}


			bool execute_case(const std::string& parent, int i0, int n0, base_test_case* tcase)
			{
				std::printf("[%s %d/%d] test case: %s\n",
						parent.c_str(), i0, n0, tcase->name().c_str());

				try
				{
					tcase->run();

					return true;
				}
				catch(assertion_failure& exc)
				{
					std::printf("Assertion Failure at {%s (%d)}: %s\n",
							exc.filename().c_str(), exc.line_number(), exc.message().c_str());
				}

				return false;
			}


		}; // end class stdc_execution_monitor

	}
}


// Macros to facilitate writing main test file

#define BCS_TEST_MAIN_FUNCTION \
	int main(int argc, char *argv[]) { \
		bcs::test::stdc_execution_monitor exec_mon; \
		bcs::test::test_suite *tsuite = master_suite(); \
		exec_mon.execute(tsuite); \
		delete tsuite; \
		return 0; \
	}



#endif 


