import unittest
import pyscf.gto
from pyscf import scf
from SCF_DIIS import SCF

class SCFTester(unittest.TestCase):

   #RHF ---------------

    def test1_H2_sto3g(self):
        print('\nSetting up H2 with sto-3g basis...')
        self.H2 = pyscf.gto.M(verbose = 0, atom = [ ['H', (0.0, 0.0, 0.0)], ['H', (0.74, 0.0, 0.0)], ], basis = 'sto-3g', symmetry = True)
        self.SCF = SCF(self.H2)
        self.SCF.kernel()
        self.assertAlmostEqual(-1.11675930739643, self.SCF.e_tot, places=7)
        print('H2 at sto-3g agrees with PySCF!')

    def test3_H2_cc_PVDZ(self):
        print('\nSetting up H2 with cc-PVDZ basis...')
        self.H2 = pyscf.gto.M(verbose = 0, atom = [ ['H', (0.0, 0.0, 0.0)], ['H', (0.74, 0.0, 0.0)], ], basis = 'cc-PVDZ', symmetry = True)
        self.SCF = SCF(self.H2)
        self.SCF.kernel()
        self.assertAlmostEqual(-1.12870009355644, self.SCF.e_tot, places=7)
        print('H2 at cc-PVDZ agrees with PySCF!')
    
    #N2 Tests -------------------

    def test4_N2_sto3g(self):
        print('\nSetting up N2 with sto-3g basis...')
        self.N2 = pyscf.gto.M(verbose = 0, atom = [ ['N', (0.0, 0.0, 0.0)], ['N', (1.1, 0.0, 0.0)], ], basis = 'sto-3g', symmetry = True)
        self.SCF = SCF(self.N2)
        self.SCF.kernel()
        self.assertAlmostEqual(-107.49650051179789, self.SCF.e_tot, places=7)
        print('N2 at sto-3g agrees with PySCF!')
    def test5_N2_631g(self):
        print('\nSetting up N2 with 6-31g basis...')
        self.N2 = pyscf.gto.M(verbose = 0, atom = [ ['N', (0.0, 0.0, 0.0)], ['N', (1.1, 0.0, 0.0)], ], basis = '6-31g', symmetry = True)
        self.SCF = SCF(self.N2)
        self.SCF.kernel()
        self.assertAlmostEqual(-108.867618373058, self.SCF.e_tot, places=7)
        print('N2 at 6-31g agrees with PySCF!')
    def test6_N2_cc_PVDZ(self):
        print('\nSetting up N2 with cc-PVDZ basis...')
        self.N2 = pyscf.gto.M(verbose = 0, atom = [ ['N', (0.0, 0.0, 0.0)], ['N', (1.1, 0.0, 0.0)], ], basis = 'cc-PVDZ', symmetry = True)
        self.SCF = SCF(self.N2)
        self.SCF.kernel()
        self.assertAlmostEqual(-108.953796240891, self.SCF.e_tot, places=7)
        print('N2 at cc-PVDZ agrees with PySCF!')
   
   #CH4 Tests

    def test7_CH4_sto3g(self):
        print('\nSetting up CH4 with sto-3g basis...')
        self.CH4 = pyscf.gto.M(verbose = 0, atom = [ ['C', (0.0, 0.0, 0.0)], ['H', (0.629118, 0.629118, 0.629118)], ['H', (-0.629118, -0.629118, 0.629118)],['H', (-0.629118, 0.629118, -0.629118)], ['H', (0.629118, -0.629118, -0.629118)], ], basis = 'sto-3g', symmetry = True)
        self.SCF = SCF(self.CH4)
        self.SCF.kernel()
        self.assertAlmostEqual(-39.726715311543, self.SCF.e_tot, places=7)
        print('CH4 at sto-3g agrees with PySCF!')

    def test8_CH4_6_31g(self):
        print('\nSetting up CH4 with 6-31g basis...')
        self.CH4 = pyscf.gto.M(verbose = 0, atom = [ ['C', (0.0, 0.0, 0.0)], ['H', (0.629118, 0.629118, 0.629118)], ['H', (-0.629118, -0.629118, 0.629118)],['H', (-0.629118, 0.629118, -0.629118)], ['H', (0.629118, -0.629118, -0.629118)], ], basis = '6-31g', symmetry = True)
        self.SCF = SCF(self.CH4)
        self.SCF.kernel()
        self.assertAlmostEqual(-40.1803987599616, self.SCF.e_tot, places=7)
        print('CH4 at 6-31g agrees with PySCF!')

    def test90_CH4_cc_PVDZ(self):
        print('\nSetting up CH4 with cc-PVDZ basis with DIIS...')
        self.CH4 = pyscf.gto.M(verbose = 0, atom = [ ['C', (0.0, 0.0, 0.0)], ['H', (0.629118, 0.629118, 0.629118)], ['H', (-0.629118, -0.629118, 0.629118)],['H', (-0.629118, 0.629118, -0.629118)], ['H', (0.629118, -0.629118, -0.629118)], ], basis = 'cc-PVDZ', symmetry = True)
        self.SCF = SCF(self.CH4, DIIS=True, verbose=2)
        self.SCF.kernel()
        self.assertAlmostEqual(-40.1987085424813, self.SCF.e_tot, places=7)
        print('CH4 at cc-PVDZ agrees with PySCF!')

    def test91_CH4_cc_PVDZ(self):
        print('\nSetting up CH4 with cc-PVDZ no DIIS basis...')
        self.CH4 = pyscf.gto.M(verbose = 0, atom = [ ['C', (0.0, 0.0, 0.0)], ['H', (0.629118, 0.629118, 0.629118)], ['H', (-0.629118, -0.629118, 0.629118)],['H', (-0.629118, 0.629118, -0.629118)], ['H', (0.629118, -0.629118, -0.629118)], ], basis = 'cc-PVDZ', symmetry = True)
        self.SCF = SCF(self.CH4, DIIS=False, verbose=2)
        self.SCF.kernel()
        self.assertAlmostEqual(-40.1987085424813, self.SCF.e_tot, places=7)
        print('CH4 at cc-PVDZ agrees with PySCF!')

   #-------------------Testing UHF --------------------------#
    def test92_H2_321g_T(self):
        print('\nSetting up Triplet H2 with 3-21g basis...')
        self.H2 = pyscf.gto.M(verbose = 0, atom = [ ['H', (0.0, 0.0, 0.0)], ['H', (0.74, 0.0, 0.0)], ], basis = '3-21g', symmetry = True)
        self.SCF = SCF(self.H2, spin=2, method_type='uhf')
        self.SCF.kernel()
        self.assertAlmostEqual( -0.747716247643668 , self.SCF.e_tot, places=7)
        print('Triplet UHF H2 at 3-21g agrees with PySCF!')

    def test93_O2_321g_T(self):
        print('\nSetting up Triplet H2 with 3-21g basis...')
        self.O2 = pyscf.gto.M(verbose = 0, atom = [ ['O', (0.0, 0.0, 0.0)], ['O', (1.2, 0.0, 0.0)], ], basis = '3-21g', symmetry = True)
        self.SCF = SCF(self.O2, spin=2, method_type='uhf')
        self.SCF.kernel()
        self.assertAlmostEqual(-148.766722176946, self.SCF.e_tot, places=7)
        print('Triplet UHF O2 at 3-21g agrees with PySCF!')
   

if __name__ == '__main__':
    unittest.main()
