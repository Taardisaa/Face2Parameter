"""Independent expected arithmetic/branch/register cases for DXBC execution."""
import unittest
import numpy as np
from src.render.dxbc import Program, UnsupportedShader

class DXBC(unittest.TestCase):
    def execute(self, code, values):
        return Program('ps_4_0\n'+code).execute({'v0':np.asarray(values,np.float32)},{}).outputs['o0']

    def test_branch_and_bit_integer_dither_indices(self):
        output=self.execute('''ftoi r0.xyzw, v0.xyzw
and r0.x, r0.x, l(3)
ishl r0.x, r0.x, l(2)
iadd r0.x, r0.x, l(1)
itof r0.x, r0.x
lt r1.x, v0.x, l(2.000000)
if_nz r1.x
mov o0.x, r0.x
else
mov o0.x, l(100.000000)
endif
mov o0.yzw, l(0.000000, 1.000000, 2.000000, 3.000000)
ret''',[[1.9,0,0,0],[3.1,0,0,0]])
        np.testing.assert_array_equal(output,[[5,1,2,3],[100,1,2,3]])

    def test_sampling_shadow_pow_and_partial_swizzles(self):
        p=Program('''ps_4_0
dcl_constantbuffer CB0[7], immediateIndexed
mad r0.xy, v1.xyxx, cb0[5].xyxx, cb0[5].zwzz
sample r0.xyzw, r0.xyxx, t0.xyzw, s0
log r0.x, r0.w
mul r0.x, r0.x, cb0[6].x
exp r0.x, r0.x
min r0.x, r0.x, l(1.000000)
mul r0.x, r0.x, cb0[4].w
add r0.yzw, cb0[4].xxyz, l(0.000000,-1.000000,-1.000000,-1.000000)
mad o0.xyz, r0.xxxx, r0.yzwy, l(1.000000,1.000000,1.000000,0.000000)
mov o0.w, l(1.000000)
ret''')
        c=np.zeros((7,4),np.float32);c[4]=[.2,.3,.4,.8];c[5]=[1,1,0,0];c[6,0]=.5
        alpha=np.array([.25,.81],np.float32)
        sampled=np.ones((2,4),np.float32);sampled[:,3]=alpha
        output=p.execute({'v1':np.zeros((2,4),np.float32)},{0:c},
                         sample=lambda *args:sampled).outputs['o0']
        expected=1+(np.sqrt(alpha)*.8)[:,None]*(c[4,:3]-1)
        np.testing.assert_allclose(output[:,:3],expected,atol=1e-7,rtol=0)
        np.testing.assert_array_equal(output[:,3],1)

    def test_discard_not_resurrected_by_endif(self):
        p=Program('''ps_4_0
lt r0.x, v0.x, l(0.500000)
if_nz r0.x
discard_nz r0.x
endif
mov o0.xyzw, l(1.000000,1.000000,1.000000,1.000000)
ret''')
        result=p.execute({'v0':np.array([[.1,0,0,0],[.9,0,0,0]],np.float32)},{})
        np.testing.assert_array_equal(result.discarded,[True,False])
        np.testing.assert_array_equal(result.outputs['o0'][:,0],[0,1])

    def test_missing_resource_is_failure(self):
        p=Program('ps_4_0\nsample o0.xyzw, v0.xyzw, t0.xyzw, s0\nret')
        with self.assertRaises(UnsupportedShader):
            p.execute({'v0':np.zeros((1,4),np.float32)}, {})

    def test_float_saturation_is_not_integer_conversion(self):
        output=self.execute('mov_sat o0.xyzw, v0.xyzw\nret',[[-3,.2,1.1,2]])
        np.testing.assert_array_equal(output,[[0,np.float32(.2),1,1]])

    def test_divergent_loop_and_hex_bits(self):
        output=self.execute('''ftoi r1.x, v0.x
mov r0.x, l(0)
mov o0.xyzw, l(0,0,0,0)
loop
ige r2.x, r0.x, r1.x
breakc_nz r2.x
add o0.x, o0.x, l(1.000000)
iadd r0.x, r0.x, l(1)
endloop
mov o0.y, l(0x3f800000)
ret''',[[2,0,0,0],[5,0,0,0],[0,0,0,0]])
        np.testing.assert_array_equal(output[:,0],[2,5,0])
        np.testing.assert_array_equal(output[:,1],1)
