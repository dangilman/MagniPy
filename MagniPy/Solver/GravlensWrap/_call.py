import subprocess

def call_lensmodel(inputfile='', path_2_lensmodel=''):

    proc = subprocess.Popen([path_2_lensmodel + 'lensmodel', str(inputfile)])

    proc.wait()

    #proc = subprocess.Popen(['rm',str(inputfile)])