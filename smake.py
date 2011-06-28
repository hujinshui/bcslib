#!/usr/bin/env python

# This is a simple script to generate makefile for different platforms
#
#	This script tries to generate a makefile as simple as possible
#	so as to avoid possible problems arising in auto-build tools
#

# Copyright (c) 2011 Dahua Lin, MIT

import platform
import os
import os.path
import sys
import subprocess as sp
import ConfigParser as cp
import re

# Auxiliary functions

def dequote(sval):
	"""Dequote a string value"""
	
	if len(sval) >= 2 and (sval[0]=='"' and sval[-1]=='"'):
		return sval[1:-1]
	else:
		return sval
	

def report_err(message):
	sys.stderr.write(message + '\n')
	sys.exit(1)


def get_cmd_output(cmd):
	"""Get the stdout and stderr output of a command"""
	
	p = sp.Popen(cmd, shell=True, 
		stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
		
	return p.communicate()
	
def found_sys_command(name):
	"""Test whether a particular command is available in search path"""
	
	rout, rerr = get_cmd_output('which ' + name)
	return len(rout) > 0
	
def ver2tuple(vstr):
	"""Convert version string to tuple"""
	
	vns = tuple(int(x) for x in vstr.strip().split('.'))
	if len(vns) > 3:
		vns = vns[:3]
	elif len(vns) == 2:
		vns = (vns[0], vns[1], 0)
	elif len(vns) == 1:
		vns = (vns[0], 0, 0)
		
	return vns
	
# compiler version retrieval	
	
def get_compiler_version(name):
	"""Get the tuple that represents the version of a compiler"""
	
	if name == 'g++':
		rout, rerr = get_cmd_output('g++ -dumpversion')
		if len(rout) > 0:
			return ver2tuple(rout)
		else:
			report_err('Failed to get the version of g++')
			
	elif name == 'clang++':
		rout, rerr = get_cmd_output('clang++ --version')
		if len(rout) > 0:
			terms = rout.strip().split()
			if len(terms) >= 3:
				return ver2tuple(terms[2])
				
		report_err('Failed to get the version of clang++')
			
	else:
		report_err('Unsupported compiler name ' + name)
			
			
# parse ini file

def test_supports(supps, plf, compiler_name, compiler_ver):
	"""test whether the current compiler is supported for a particular section"""
	
	for s in supps:
		parts = s.strip().split(':')
		if len(parts) == 1:
			a = parts[0]
			cn = None
			cv = None
		elif len(parts) == 2:
			a = parts[0]
			cn = parts[1]
			cv = None
		elif len(parts) == 3:
			a = parts[0]
			cn = parts[1]
			cv = ver2tuple(parts[2])
		else:
			report_err('The support string %s is invalid' % s.strip())
			
		if a.lower() == plf.lower(): # platform match
			if cn == None:
				return True 	# no specific requirement on compiler
			else:
				# check compiler
				if cn.lower() == compiler_name.lower():
					if cv == None:
						return True
					else:
						# check compiler version
						if cv <= compiler_ver:
							return True
							
	return False
			
			
def parse_cfg(filename, plf, compiler_name, compiler_ver):
	"""parse init file and make a variable context for makefile generation"""
	
	cfg = cp.ConfigParser()
	cfg.read(filename)
	
	vdict = cfg.defaults()
	
	for sec in cfg.sections():
		if not cfg.has_option(sec, '_supports'):
			print 'WARNING: The section %s does not has a _supports option, ignored.'
			continue
			
		supps = cfg.get(sec, '_supports')
		supps = dequote(supps.strip()).split(';')
		
		if test_supports(supps, plf, compiler_name, compiler_ver):
			print 'section %s included' % sec
			
			for name, val in cfg.items(sec):
				if len(name) > 0 and name[0] != '_':
					vdict[name] = dequote(val.strip())
					
	return vdict
	
	
def generate_file(vdict, templatefile, outputfile):
	"""generate a file from a template file"""
	
	pat = re.compile('@\w[\w\d_]+@')
	
	# read from template
	with open(templatefile, 'r') as fin:
		lines = fin.readlines()
		
	i = 0
	for line in lines:
		i = i + 1
		
		sl = line.strip()
		if len(sl) == 0 or sl[0] == '#':  # empty line or comment line
			continue
		
		ms = re.findall(pat, line)
		if len(ms) > 0:
			for m in ms:
				vname = m[1:-1]
				if vname in vdict:
					vval = vdict[vname]
					line = line.replace(m, vval)
				else:
					print "WARNING: variable %s in line %d does not find a value" % (vname, i)
		
			lines[i-1] = line
		
	# write to destination
	with open(outputfile, 'w') as fout:
		for line in lines:
			fout.write(line)
	
	
			
# main procedure

if __name__ == '__main__':
	
	print "smake started ..."
	
	# detect platform
	
	os_name = platform.system()
	arch = platform.architecture()
	arch_bits = arch[0]
	plf = "%s.%s" % (os_name, arch_bits)
	
	print "Detected platform:", plf
	
	# detect compiler
	
	cxx = os.getenv('CXX')
	if cxx == None:
		if found_sys_command('g++'):
			cxx = 'g++'
		elif found_sys_command('clang++'):
			cxx = 'clang++'
		else:
			report_err('Unable to detect a supported compiler')
			
	# detect compiler version
	
	ver = get_compiler_version(cxx)
	vstr = '.'.join(str(x) for x in ver)
			
	print "Detected compiler: %s %s" % (cxx, vstr)
	
	# parse ini file
	
	cfgfile = 'smake.ini'
	if not os.path.isfile(cfgfile):
		report_err('The configuration file %s is not found.' % cfgfile)
		
	vdict = parse_cfg(cfgfile, plf, cxx, ver)
	
	if len(vdict) > 0:
		print "Applicable macros:"
		for name, val in vdict.items():
			print '    ', name, '=', val
	else:
		print "No appliable macros"
		
	vdict['cxx'] = cxx
		
	# generate makefile
		
	infile = 'makefile.in'
	outfile = 'makefile.out'
	
	if not os.path.isfile(infile):
		report_err('The input template %s is not found.' % infile)
	
	generate_file(vdict, infile, outfile)
		
	print "%s is generated" % outfile	
	print "smake done!\n"

