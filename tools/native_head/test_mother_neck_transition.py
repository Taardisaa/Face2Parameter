import unittest

import numpy as np

from tools.native_head.mother_neck_transition import collar


class CollarContract(unittest.TestCase):
    def setUp(self):
        angle=np.arange(12)*2*np.pi/12
        self.original=np.concatenate([np.column_stack([
            (1+.05*k)*np.cos(angle), np.full(12,k*.2),
            (1+.05*k)*np.sin(angle)]) for k in range(10)])
        self.faces=np.array([[k*12+i,k*12+(i+1)%12,(k+1)*12+i]
            for k in range(9) for i in range(12)]+[
            [k*12+(i+1)%12,(k+1)*12+(i+1)%12,(k+1)*12+i]
            for k in range(9) for i in range(12)])
        self.lift=np.array([0.,.12,0.])

    def test_native_lower_derivative_and_rigid_feature_are_preserved(self):
        authored=self.original.copy()
        authored[:,2]+=.3
        protected=np.zeros(len(authored),bool); protected[60:63]=True
        result,d=collar(self.original,authored,self.faces,np.arange(12),
                        protected,self.lift)
        np.testing.assert_array_equal(result[d['lower']],self.original[d['lower']])
        np.testing.assert_array_equal(result[d['upper']],authored[d['upper']]+self.lift)
        np.testing.assert_array_equal(result[:12]-result[12:24],
                                      self.original[:12]-self.original[12:24])

    def test_no_edit_without_shape_or_placement_change(self):
        result,_=collar(self.original,self.original,self.faces,np.arange(12),
                       np.zeros(120,bool),np.zeros(3))
        np.testing.assert_array_equal(result,self.original)

    def test_protected_interface_conflict_is_rejected(self):
        protected=np.zeros(120,bool);protected[12]=True
        with self.assertRaisesRegex(ValueError,'protected feature'):
            collar(self.original,self.original,self.faces,np.arange(12),
                   protected,self.lift)


if __name__=='__main__':
    unittest.main()
