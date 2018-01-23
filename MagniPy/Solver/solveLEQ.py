from GravlensWrap.generate_input import *
import subprocess

def lensmodel_call(inputfile='',path_2_lensmodel=''):

    proc = subprocess.Popen([path_2_lensmodel + 'lensmodel', str(inputfile)])

    proc.wait()

