import unittest
import numpy as np
from src.render.native_inputs import pack_sh,sss_pass_schedule,view_format
from src.render.dxbc import UnsupportedShader

class NativeInputs(unittest.TestCase):
    def test_sh_has_full_quadratic_terms(self):
        sh=np.arange(27,dtype=np.float64).reshape(3,9)
        packed=pack_sh(sh)
        for normal in ([1,0,0],[0,1,0],[0,0,1],[.6,.8,0]):
            x,y,z=normal
            direct=sh[:,0]+sh[:,1]*y+sh[:,2]*z+sh[:,3]*x+sh[:,4]*x*y+sh[:,5]*y*z+sh[:,6]*(3*z*z-1)+sh[:,7]*x*z+sh[:,8]*(x*x-y*y)
            a=np.array([packed['unity_SHA'+c]@[*normal,1] for c in 'rgb'])
            b=np.array([packed['unity_SHB'+c]@[x*y,y*z,z*z,z*x] for c in 'rgb'])
            computed=a+b+packed['unity_SHC'][:3].astype(np.float64)*(x*x-y*y)
            np.testing.assert_allclose(computed,direct,atol=3e-6,rtol=0)

    def test_sss_schedule_preserves_source_iterations_and_radius(self):
        s={'<ScatteringIterations>k__BackingField':2,'<ScatteringRadius>k__BackingField':1,'<Downsampling>k__BackingField':2}
        result=sss_pass_schedule(s,{'initFOV':23},width=1600,height=900,current_fov=23,stereo=False)
        self.assertEqual([r['axis'] for r in result],['vertical','horizontal','vertical','horizontal'])
        self.assertAlmostEqual(float(result[0]['_TexelOffsetScale'][1]),.41666666,places=6)

    def test_no_typeless_gamma_guess(self):
        with self.assertRaises(UnsupportedShader):
            view_format({'format':'RGBA32'},{'dxgi_format':27})
        self.assertEqual(view_format({'format':'ARGB32','srgb':True},{'dxgi_format':27}),29)
